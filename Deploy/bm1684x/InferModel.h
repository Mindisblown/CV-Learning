#include "opencv2/opencv.hpp"
#include <bmruntime_cpp.h>

using namespace bmruntime;

class InferModel
{
public:
    InferModel();
    ~InferModel();
private:
    int Init(const char *model_path, const char *net_name);
    int Preprocess(cv::Mat &inpput_data, int data_idx); //包含预处理
    int Inference();
    Context model_ctx_; //网络管理-加载模型获取网络信息等
    bm_handle_t algo_handle_; //设备句柄
    Network *model_net_; //实际的网络
    bm_tensor_t *input_tensor_;
    bm_tensor_t *output_tensor_;

    std::vector<bm_image> resize_imgs_; //resize数据
    std::vector<bm_image> convert_imgs_;
    std::vector<bm_image> infer_imgs_; //最多容纳的推理图片buff

    bm_device_mem_t resize_device_mem_, convert_device_mem_, infer_device_mem_; //填充bm_image需要的device memory

    bmcv_convert_to_attr preprocess_attr_; //预处理参数

    unsigned long long infer_buff_addr_, out_buff_addr_, resize_buff_addr_;
    //模型输出大小
    int net_input_h_, net_input_w_, net_input_c_;
    int net_output_h_, net_output_w_, net_output_c_;
    int single_img_length_, single_channel_length_;

    bool algo_init_state_;

    int max_batch_ = 4;
    int max_infer_buffer_ = 512;

}