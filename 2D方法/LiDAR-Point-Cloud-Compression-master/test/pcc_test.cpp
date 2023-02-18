/* this is an example of 3d lidar compression.
 * */

#include "pcc_module.h"
#include "encoder.h"
#include "decoder.h"
#include "io.h"
#include "time.h"

 #include <boost/program_options/parsers.hpp>

#include <boost/program_options/variables_map.hpp>
#include <boost/program_options/options_description.hpp>

using namespace std;
using namespace cv;

//计算PSNR和SSIM

double getPSNR(const Mat& I1, const Mat& I2)
{
    Mat s1;
    absdiff(I1, I2, s1);       // |I1 - I2|
    s1.convertTo(s1, CV_32F);  // cannot make a square on 8 bits
    s1 = s1.mul(s1);           // |I1 - I2|^2
    Scalar s = sum(s1);         // sum elements per channel
    double sse = s.val[0] + s.val[1] + s.val[2]; // sum channels
    if( sse <= 1e-10) // for small values return zero
        return 0;
    else
    {
        double  mse =sse /(double)(I1.channels() * I1.total());
        double psnr = 10.0*log10((255*255)/mse);
        return psnr;
    }
}
Scalar getMSSIM( const Mat& i1, const Mat& i2)
{
    const double C1 = 6.5025, C2 = 58.5225;
    /***************************** INITS **********************************/
    int d     = CV_32F;
    Mat I1, I2;
    i1.convertTo(I1, d);           // cannot calculate on one byte large values
    i2.convertTo(I2, d);
    Mat I2_2   = I2.mul(I2);        // I2^2
    Mat I1_2   = I1.mul(I1);        // I1^2
    Mat I1_I2  = I1.mul(I2);        // I1 * I2
    /*************************** END INITS **********************************/
    Mat mu1, mu2;   // PRELIMINARY COMPUTING
    GaussianBlur(I1, mu1, Size(11, 11), 1.5);
    GaussianBlur(I2, mu2, Size(11, 11), 1.5);
    Mat mu1_2   =   mu1.mul(mu1);
    Mat mu2_2   =   mu2.mul(mu2);
    Mat mu1_mu2 =   mu1.mul(mu2);
    Mat sigma1_2, sigma2_2, sigma12;
    GaussianBlur(I1_2, sigma1_2, Size(11, 11), 1.5);
    sigma1_2 -= mu1_2;
    GaussianBlur(I2_2, sigma2_2, Size(11, 11), 1.5);
    sigma2_2 -= mu2_2;
    GaussianBlur(I1_I2, sigma12, Size(11, 11), 1.5);
    sigma12 -= mu1_mu2;
    Mat t1, t2, t3;
    t1 = 2 * mu1_mu2 + C1;
    t2 = 2 * sigma12 + C2;
    t3 = t1.mul(t2);              // t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))
    t1 = mu1_2 + mu2_2 + C1;
    t2 = sigma1_2 + sigma2_2 + C2;
    t1 = t1.mul(t2);               // t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))
    Mat ssim_map;
    divide(t3, t1, ssim_map);      // ssim_map =  t3./t1;
    Scalar mssim = mean( ssim_map ); // mssim = average of ssim map
    return mssim;
}





int main(int argc, char** argv) { 
  float start = clock();//开始时间
  
  std::string file_path;
  std::string input_format("binary");
  float pitch_precision, yaw_precision, threshold;
  int tile_size;

  namespace po = boost::program_options;

  po::options_description opts("PCC options");
  opts.add_options()
    ("help,h", "Print help messages")
    ("file", po::value<std::string>(&file_path)->required(), "raw point cloud data path")
    ("pitch,p", po::value<float>(&pitch_precision)->required(), "pitch precision")
    ("yaw,y", po::value<float>(&yaw_precision)->required(), "yaw precision")
    ("threshold,t", po::value<float>(&threshold)->required(), "threshold value for fitting")
    ("format,f",  po::value<std::string>(&input_format),
     "trace_file input format: binary(default) or ascii")
    ("tile,l", po::value<int>(&tile_size)->required(), "fitting tile size");

  po::variables_map vm;

  try 
  {
    po::store(po::parse_command_line(argc, argv, opts), vm);
    
    if (vm.count("help")) 
    {
      std::cout << "Point Cloud Compression" << std::endl 
        << opts << std::endl;
      return 0;
    }

    po::notify(vm);
  } catch(po::error& e) { 
    std::cerr << "ERROR: " << e.what() << std::endl << std::endl; 
    std::cerr << opts << std::endl; 
    return -1;
  }
  
  // create a vector to store frames;
  std::vector<point_cloud> pcloud_data;
  load_pcloud(file_path, pcloud_data);

  PccResult pcc_res;

  /*******************************************************************/
  // initialization

  int row = (VERTICAL_DEGREE/yaw_precision);
  row = ((row + tile_size-1)/tile_size)*tile_size;
  int col = HORIZONTAL_DEGREE/pitch_precision + tile_size;
  col = ((col + tile_size-1)/tile_size)*tile_size;

  double proj_time, fit_time;
  float psnr, total_pcloud_size;

  /*******************************************************************/
  // convert range map

  std::cout << "CURRENT pcloud size: " << pcloud_data.size() << std::endl;
  
  // Characterize Range Map
  // floating map;
  cv::Mat* f_mat = new cv::Mat(row, col, CV_32FC4, cv::Scalar(0.f,0.f,0.f,0.f));
  
  proj_time = map_projection(*f_mat, pcloud_data, pitch_precision, yaw_precision, 'e');


  pcc_res.proj_times->push_back(proj_time);
  
  // compute compression rate: bit-per-point (bpp)
  pcc_res.compression_rate->push_back(8.0f*f_mat->cols*f_mat->rows/pcloud_data.size());
  
  // loss error compute;
  //psnr = compute_loss_rate<cv::Vec4f>(*f_mat, pcloud_data, pitch_precision, yaw_precision);
  
  // update the info;
  pcc_res.loss_rate->push_back(psnr);
  
  std::cout << "Loss rate [PSNR]: " << psnr << " Compression rate: "
            << pcc_res.compression_rate->back() << " bpp." << std::endl;
  
  /*******************************************************************/
  // fitting range map
  int mat_div_tile_sizes[] = {row/tile_size, col/tile_size};
  std::vector<cv::Vec4f> coefficients;
  std::vector<int> tile_fit_lengths;
  std::vector<float> unfit_nums;

  cv::Mat* b_mat = new cv::Mat(row/tile_size, col/tile_size, CV_32SC1, 0.f);
  cv::Mat* occ_mat = new cv::Mat(row/tile_size, col/tile_size, CV_32SC1, 0.f);
  
  // encode the occupatjon map  
  encoder::encode_occupation_mat(*f_mat, *occ_mat, tile_size, mat_div_tile_sizes);

  fit_time = encoder::single_channel_encode(*f_mat, *b_mat, mat_div_tile_sizes, coefficients, 
                                            unfit_nums, tile_fit_lengths,
                                            threshold, tile_size);
 
  pcc_res.fit_times->push_back(fit_time);

  // what we need to store:
  // 1. b_mat: binary map for tile type
  export_b_mat(*b_mat, "b_mat.bin");
  delete b_mat;

  b_mat = new cv::Mat(row/tile_size, col/tile_size, CV_32SC1, 0.f);
  // 2. planar coefficients
  export_coefficients(coefficients, "coefficients.bin");
  coefficients.clear();
  
  // 3. occ_mat: occupation map
  export_occ_mat(*occ_mat, "occ_mat.bin");
  delete occ_mat;

  occ_mat = new cv::Mat(row/tile_size, col/tile_size, CV_32SC1, 0.f);
  // 4. unfit_nums: unfitted_nums
  export_unfit_nums(unfit_nums, "unfit_nums.bin");
  unfit_nums.clear();
  
  // 5. tile_fit_lengths
  export_tile_fit_lengths(tile_fit_lengths, "tile_fit_lengths.bin");
  tile_fit_lengths.clear();
  

  system("tar -cvzf frame.tar.gz *.bin");
  system("tar -xvzf frame.tar.gz");

  import_b_mat(*b_mat, "b_mat.bin");
  import_coefficients(coefficients, "coefficients.bin");
  import_occ_mat(*occ_mat, "occ_mat.bin");
  import_unfit_nums(unfit_nums, "unfit_nums.bin");
  import_tile_fit_lengths(tile_fit_lengths, "tile_fit_lengths.bin");
  // reconstruct the range image
  cv::Mat* r_mat = new cv::Mat(row, col, CV_32FC1, 0.f);
  // decoding
  decoder::single_channel_decode(*r_mat, *b_mat, mat_div_tile_sizes, coefficients, 
                                 *occ_mat, tile_fit_lengths, unfit_nums, tile_size);

  psnr = compute_loss_rate(*r_mat, pcloud_data, pitch_precision, yaw_precision);
    
  std::vector<point_cloud> restored_pcloud;
  restore_pcloud(*r_mat, pitch_precision, yaw_precision, restored_pcloud);
  
  cv::Mat* f_mat2 = new cv::Mat(row, col, CV_32FC1, 0.f);
  pcloud_to_mat<float>(restored_pcloud, *f_mat2, pitch_precision, yaw_precision);
  
  psnr = compute_loss_rate(*r_mat, restored_pcloud, pitch_precision, yaw_precision);
 
  // output_cloud(pcloud_data, "org.ply");
  // output_cloud(restored_pcloud, "restored.ply");
  std::cout << "**********************************************************" << std::endl;
  //std::cout << "PSNR" << std::endl;
  //print_pcc_res(pcc_res);



//画拟合前后的二维图像

  std::vector<cv::Mat> mv;
  cv::split(*f_mat, mv);   //通道分离  split函数用于将一个多通道数组分离成几个单通道数组。
  //imshow("channel0", mv[0]);
  //cv::waitKey(0);

  double minv = 0.0, maxv = 0.0;
  double* minp = &minv;
  double* maxp = &maxv;
  cv::minMaxIdx(mv[0],minp,maxp);

  cv::Mat range_mat;
  range_mat=(mv[0])/(*maxp)*255;
  //std::cout<<range_mat<<std::endl;

  cv::Mat tep;
  range_mat.convertTo(tep, CV_8UC1, 255.0/255);
  //std::cout<<tep<<std::endl;
  bool result = cv::imwrite("ini.png", tep);
  cv::imshow("tep", tep);
  cv::waitKey(0);

/*
  double minv = 0.0, maxv = 0.0;
  double* minp = &minv;
  double* maxp = &maxv;
  cv::minMaxIdx(*r_mat,minp,maxp);

  cv::Mat range_mat;
  range_mat=(*r_mat)/(*maxp)*255;
  //std::cout<<range_mat<<std::endl;

  cv::Mat tep;
  range_mat.convertTo(tep, CV_8UC1, 255.0/255);
  //std::cout<<tep<<std::endl;
  bool result = cv::imwrite("after_compression.png", tep);
  cv::imshow("tep", tep);
  cv::waitKey(0);


  //std::cout << "Mat minv = " << minv << std::endl;
  //std::cout << "Mat maxv = " << maxv << std::endl;
  //std::cout<<*r_mat<<std::endl;
  //cv::imshow("r_mat", *r_mat);
  //cv::waitKey(0);*/


  //计算压缩前后的2D图像的PSNR和SSIM
  /*
  std::vector<cv::Mat> mv;
  cv::split(*f_mat, mv);   //通道分离  split函数用于将一个多通道数组分离成几个单通道数组。
  double compare_panr=getPSNR(mv[0],*r_mat);
  Scalar compare_ssim=getMSSIM(mv[0],*r_mat);
  std::cout << "compare_panr = " << compare_panr << std::endl;
  std::cout << "compare_ssim = " << compare_ssim << std::endl;*/


  
  delete f_mat; //原始深度图像
  delete r_mat; //重构后的深度图像
  delete occ_mat;
  delete f_mat2;

  float end = clock();//结束时间
 
  float TimeValue;
  TimeValue=end-start;	
  std::cout << "time = " << TimeValue/CLOCKS_PER_SEC << std::endl;
  return 0;
}

