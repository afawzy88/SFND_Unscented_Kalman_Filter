#include "ukf.h"
#include "Eigen/Dense"
#include "iostream"

using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF()
{
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 2.5;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = M_PI/4;
  
  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  
  /**
   * End DO NOT MODIFY section for measurement noise values 
   */
  
  /**
   * TODO: Complete the initialization. See ukf.h for other member properties.
   * Hint: one or more values initialized above might be wildly off...
   */

  // State dimension
  n_x_ = 5;

  // Augmented state dimension
  n_aug_ = 7;

  // Sigma point spreading parameter
  lambda_ = 3 - n_aug_;

  // Set weights
  weights_ = VectorXd(2 * n_aug_ + 1);

  double weight_0 = lambda_/(lambda_+n_aug_);
  double weight = 0.5/(lambda_+n_aug_);

  weights_(0) = weight_0;
  for (int i = 1; i < 2 * n_aug_ + 1; ++i)
  {  
    weights_(i) = weight;
  }


  // initially set to false, set to true in first call of ProcessMeasurement
  is_initialized_ = false;

  // time when the state is true, in us
  time_us_ = 0;

  // Square root of P matrix
  A = MatrixXd(n_x_, n_x_);

  // Augmented mean vector
  x_aug = VectorXd(n_aug_);

  // Augmented state covariance
  P_aug = MatrixXd(n_aug_, n_aug_);

  // Augmented sigma point matrix
  Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);

  // Process covariance matrix
  Q = MatrixXd(n_aug_-n_x_, n_aug_-n_x_);

  // Predicted sigma points matrix
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

  NISLidar = 0;
  NISRadar = 0;
}

UKF::~UKF() {}

void UKF::ProcessMeasurement(MeasurementPackage meas_package)
{
  /**
   * TODO: Complete this function! Make sure you switch between lidar and radar
   * measurements.
   */
  if (!is_initialized_)
  {
    if (meas_package.sensor_type_ == MeasurementPackage::LASER)
    {
      double px_lid = meas_package.raw_measurements_(0);
      double py_lid = meas_package.raw_measurements_(1);

      // State (No info about speed or yaw from LiDAR sensor)
      x_ << px_lid,
            py_lid,
            0,  // To be tuned with some values
            0,  // To be tuned with some values
            0;  // To be tuned with some values
      
      // State Covariance initialization suggestion (Lesson 4, Section 32)
      P_ = MatrixXd::Identity(5,5);
      P_(0,0) = std_laspx_*std_laspx_;
      P_(1,1) = std_laspy_*std_laspy_;      
    }

    else if (meas_package.sensor_type_ == MeasurementPackage::RADAR)
    {
      double rho    = meas_package.raw_measurements_(0);
      double phi    = meas_package.raw_measurements_(1);
      //double rhodot = meas_package.raw_measurements_(2);

      double px_rad = rho*cos(phi);
      double py_rad = rho*sin(phi);

      x_ << px_rad,
            py_rad,
            0,  // To be tuned with some values
            0,  // To be tuned with some values
            0;  // To be tuned with some values
      
      // State Covariance initialization suggestion (Lesson 4, Section 32)
      P_ = MatrixXd::Identity(5,5);
      P_(0,0) = std_radr_*std_radr_; // To be tuned with some values
      P_(1,1) = std_radr_*std_radr_; // To be tuned with some values
      P_(2,2) = std_radr_*std_radr_; // To be tuned with some values
    }

    is_initialized_ = true;
    time_us_ = meas_package.timestamp_;
    return;
  }

  double delta_t  = (meas_package.timestamp_ - time_us_) / 1000000.0;
  time_us_ = meas_package.timestamp_;

  //==================================================================================
  // Predict
  Prediction(delta_t);
  //==================================================================================
  // Update
  if (meas_package.sensor_type_ == MeasurementPackage::LASER)
  {
    UpdateLidar(meas_package);
  } 
  else if (meas_package.sensor_type_ == MeasurementPackage::RADAR)
  {
    UpdateRadar(meas_package);
  }
  
}

void UKF::Prediction(double delta_t)
{
  /**
   * TODO: Complete this function! Estimate the object's location. 
   * Modify the state vector, x_. Predict sigma points, the state, 
   * and the state covariance matrix.
   */
  // Create augmented mean state
  VectorXd mio = VectorXd(2);
  mio.setZero();
  x_aug.head(5) = x_;
  x_aug.tail(2) = mio;

  // Create augmented covariance matrix
  Q << std_a_*std_a_,           0,
             0      , std_yawdd_*std_yawdd_; 
    
  P_aug.setZero();
  P_aug.topLeftCorner(5,5) = P_;
  P_aug.bottomRightCorner(2,2) = Q;

  // Create square root of P
  A = P_aug.llt().matrixL();

  // Create augmented sigma points
  Xsig_aug.col(0) = x_aug;

  for (int i = 0; i < n_aug_; ++i)
  {
    Xsig_aug.col(i+1)       = x_aug + sqrt(lambda_+n_aug_) * A.col(i);
    Xsig_aug.col(i+1+n_aug_) = x_aug - sqrt(lambda_+n_aug_) * A.col(i);
  }
  
  // Predict sigma points
   for (int i = 0; i < 2 * n_aug_ + 1; ++i) 
   {
    // extract values for better readability
    double p_x      = Xsig_aug(0,i);
    double p_y      = Xsig_aug(1,i);
    double v        = Xsig_aug(2,i);
    double yaw      = Xsig_aug(3,i);
    double yawd     = Xsig_aug(4,i);
    double nu_a     = Xsig_aug(5,i);
    double nu_yawdd = Xsig_aug(6,i);

   // predicted state values
    double px_p, py_p;

    // avoid division by zero
    if (fabs(yawd) > 0.001)
    {
        px_p = p_x + v/yawd * ( sin(yaw + yawd*delta_t) - sin(yaw));
        py_p = p_y + v/yawd * ( cos(yaw) - cos(yaw+yawd*delta_t) );
    } 
    else
    {
        px_p = p_x + v*delta_t*cos(yaw);
        py_p = p_y + v*delta_t*sin(yaw);
    }

    double v_p = v;
    double yaw_p = yaw + yawd*delta_t;
    double yawd_p = yawd;

    // add noise
    px_p    = px_p    + 0.5*nu_a*delta_t*delta_t * cos(yaw);
    py_p    = py_p    + 0.5*nu_a*delta_t*delta_t * sin(yaw);
    v_p     = v_p     + nu_a*delta_t;
    yaw_p   = yaw_p   + 0.5*nu_yawdd*delta_t*delta_t;
    yawd_p  = yawd_p  + nu_yawdd*delta_t;

  // write predicted sigma points into right column
    Xsig_pred_(0,i) = px_p;
    Xsig_pred_(1,i) = py_p;
    Xsig_pred_(2,i) = v_p;
    Xsig_pred_(3,i) = yaw_p;
    Xsig_pred_(4,i) = yawd_p;
   }

   // Create vector for predicted state
  VectorXd x_Pred = VectorXd(n_x_);
  x_Pred.setZero();

  // Create covariance matrix for prediction
  MatrixXd P_Pred = MatrixXd(n_x_, n_x_);
  P_Pred.setZero();

   // Predict state mean
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) 
  {  // iterate over sigma points
    x_Pred = x_Pred + weights_(i) * Xsig_pred_.col(i);
  }

  // Predict state covariance matrix
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) 
  { 
    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_Pred;
    // angle normalization
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

    P_Pred = P_Pred + weights_(i) * x_diff * x_diff.transpose() ;
  }

  x_ = x_Pred;
  P_ = P_Pred;

}

void UKF::UpdateLidar(MeasurementPackage meas_package)
{
  /**
   * TODO: Complete this function! Use lidar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the lidar NIS, if desired.
   */

  // Set measurement dimension, LiDAR can measure px and py
  int n_z = 2;

  // Create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);
  Zsig.setZero();

  // Mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  z_pred.setZero();
  
  // Measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z,n_z);
  S.setZero();

  // Radar measurement vector
  VectorXd z = VectorXd(n_z);

  // Matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);
  Tc.setZero();

  // Transform sigma points into measurement space
  double px, py;

  for (int i = 0; i < 2 * n_aug_ +1; ++i) 
  {
    px  = Xsig_pred_(0,i);
    py  = Xsig_pred_(1,i);

    Zsig(0,i) = px;
    Zsig(1,i) = py;
  }

  // Calculate mean predicted measurement
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) 
  {  // iterate over sigma points
    z_pred = z_pred + weights_(i) * Zsig.col(i);
  }

  // Calculate innovation covariance matrix S
  MatrixXd R = MatrixXd(n_z,n_z);
  R(0,0) = std_laspx_*std_laspx_;
  R(1,1) = std_laspy_*std_laspy_;

  for (int i = 0; i < 2 * n_aug_ + 1; ++i) 
  { 
    // state difference
    VectorXd z_diff = Zsig.col(i) - z_pred;
    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  S = S + R;

  z <<
     meas_package.raw_measurements_(0),   // px in m
     meas_package.raw_measurements_(1);   // py in m

  // Calculate cross correlation matrix
  for (int i = 0; i < 2 * n_aug_ +1; ++i) 
  {
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;
    
    VectorXd z_diff = Zsig.col(i) - z_pred;

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  // Calculate Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  // Update state mean and covariance matrix
   VectorXd z_diff = z - z_pred;

  x_ = x_ + K * (z_diff);
  P_ = P_ - K * S * K.transpose();

  NISLidar = z_diff.transpose() * S.inverse() * z_diff;
  //std::cout << "NISLidar =  " << NISLidar << std::endl;

}

void UKF::UpdateRadar(MeasurementPackage meas_package)
{
  /**
   * TODO: Complete this function! Use radar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the radar NIS, if desired.
   */

  // Set measurement dimension, radar can measure r, phi, and r_dot
  int n_z = 3;

  // Create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);
  Zsig.setZero();

  // Mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  z_pred.setZero();
  
  // Measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z,n_z);
  S.setZero();

  // Radar measurement vector
  VectorXd z = VectorXd(n_z);

  // Matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);
  Tc.setZero();

  // Transform sigma points into measurement space
  double radr = 0; double radphi = 0; double radrd = 0;
  double px = 0; double py = 0; double v = 0; double yaw = 0;
  for (int i = 0; i < 2 * n_aug_ +1; ++i) 
  {
    px  = Xsig_pred_(0,i);
    py  = Xsig_pred_(1,i);
    v   = Xsig_pred_(2,i);
    yaw = Xsig_pred_(3,i);

    radr    = sqrt(px*px + py*py);
    radphi  = atan2(py,px);
    radrd   = (px*v*cos(yaw) + py*v*sin(yaw))/radr;

    Zsig(0,i) = radr;
    Zsig(1,i) = radphi;
    Zsig(2,i) = radrd;
  }

  // Calculate mean predicted measurement
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) 
  {  // iterate over sigma points
    z_pred = z_pred + weights_(i) * Zsig.col(i);
  }

  // Calculate innovation covariance matrix S
  MatrixXd R = MatrixXd(n_z,n_z);
  R.setZero();
  R(0,0) = std_radr_*std_radr_;
  R(1,1) = std_radphi_*std_radphi_;
  R(2,2) = std_radrd_*std_radrd_;

  for (int i = 0; i < 2 * n_aug_ + 1; ++i) 
  { 
    // state difference
    VectorXd z_diff = Zsig.col(i) - z_pred;
    // angle normalization
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  S = S + R;

  z <<
     meas_package.raw_measurements_(0),   // rho in m
     meas_package.raw_measurements_(1),   // phi in rad
     meas_package.raw_measurements_(2);   // rho_dot in m/s

  // Calculate cross correlation matrix
  for (int i = 0; i < 2 * n_aug_ +1; ++i) 
  {
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    VectorXd z_diff = Zsig.col(i) - z_pred;
    // angle normalization
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  // Calculate Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  // Ipdate state mean and covariance matrix
   VectorXd z_diff = z - z_pred;
  // angle normalization
  while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
  while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

  x_ = x_ + K * (z_diff);
  P_ = P_ - K * S * K.transpose();

  NISRadar = z_diff.transpose() * S.inverse() * z_diff;
  //std::cout << "NISRadar =  " << NISRadar << std::endl;

}