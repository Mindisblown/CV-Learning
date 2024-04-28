#include "InferModel.h"

InferModel::InferModel() : algo_handle_(nullptr), algo_init_state_(false)
{
    std::cout << "model create" << std::endl;
}

InferModel::~InferModel()
{
    std::cout << "model destory" << std::endl;
    if(algo_init_state_)
    {
        //解除设备内存的映射
        bm_mem_unmap_device_mem(algo_handle_, (int8_t*)out_buff_addr_, bm_mem_get_device_size(output_tensor_->device_mem));
        bm_mem_unmap_device_mem(algo_handle_, (int8_t*)infer_buff_addr_, bm_mem_get_device_size(memory_device_mem_));
        bm_mem_unmap_device_mem(algo_handle_, (int8_t*)resize_buff_addr_, bm_mem_get_device_size(resize_device_mem_));

        bm_free_device(algo_handle_, convert_device_mem_);
        bm_free_device(algo_handle_, resize_device_mem_);
        bm_free_device(algo_handle_, infer_device_mem_);

        for(int i = 0; i < max_infer_buffer_; ++i)
        {
            bm_image_destory(infer_imgs_[i]);
        }
        for(int j = 0; j < max_batch_; ++j)
        {
            bm_image_destory(resize_imgs_[j]);
            bm_image_destory(convert_imgs_[j]);
        }

        delete model_net_;
    }
}

int InferModel::Init(const char *model_path, const char *net_name)
{
    bm_status_t process_status;
    process_status = model_ctx_.load_bmodel(model_path);
    assert(process_status == BM_SUCCESS);
    
    model_net_= new Network(model_ctx_, net_name);
    algo_handle_ = model_ctx_.handle();

    auto &inputs = model_net_->Inputs();
    input_tensor_ = (bm_tensor_t*)inputs[0]->tensor();
    auto &outputs = model_net->Outputs();
    output_tensor_ = (bm_tensor_t*)outputs[0]->tensor();

    net_input_c_ = input_tensor_->shape.dims[1];
    net_input_h_ = input_tensor_->shape.dims[2];
    net_input_w_ = input_tensor_->shape.dims[3];

    net_output_c_ = output_tensor_->shape.dims[1];
    net_output_h_ = output_tensor_->shape.dims[2];
    net_output_w_ = output_tensor_->shape.dims[3];

    single_img_length_ = net_input_c_ * net_input_w_ * net_input_h_;
    single_channel_length_ = net_input_w_ * net_input_h_;

    //模型的数据预处理 每个通道像素执行 y = alpha * x + beta 
    preprocess_attr_.alpha_0 = 1.0f;
    preprocess_attr_.beta_0 = 0.0f;
    preprocess_attr_.alpha_1 = 1.0f;
    preprocess_attr_.beta_1 = 0.0f;
    preprocess_attr_.alpha_2 = 1.0f;
    preprocess_attr_.beta_3 = 0.0f;

    //tensor具有4维，其维度为NCHW
    inputs[0]->Reshape({4, {1, 3, net_input_h_, net_input_w_}});

    resize_imgs_.resize(max_batch_);
    convert_imgs_.resize(max_batch_);
    infer_imgs_.resize(max_buffer_length_);

    //64对齐
    int aligned_net_w = FFALIGN(net_input_w_, 64);
    int strides[3] = {aligned_net_w, aligned_net_w, aligned_net_w};

    for(int i = 0; i < max_infer_buffer_; ++i)
    {
        process_status = bm_image_create(algo_handle, net_input_h_, net_input_w_, 
        FORMAT_RGB_PLANAR, DATA_TYPE_EXT_1N_BYTE, &infer_imgs_[i], strides);
    }

    for(int j = 0; j < max_batch_; ++j)
    {
        process_status = bm_image_create(algo_handle, net_input_h_, net_input_w_, 
            FORMAT_RGB_PLANAR, DATA_TYPE_EXT_1N_BYTE, &resize_imgs_[i], strides);

        //依据输入的类型值不同创建不同的数据格式
        if(0 == input_tensor_->dtype)
        {
            process_status = bm_image_create(algo_handle, net_input_h_, net_input_w_, 
                FORMAT_RGB_PLANAR, DATA_TYPE_EXT_FLOAT32, &convert_imgs_[i], strides);
        }
        else if(3 == input_tensor_->dtype)
        {
            process_status = bm_image_create(algo_handle, net_input_h_, net_input_w_, 
                FORMAT_RGB_PLANAR, DATA_TYPE_EXT_1N_BYTE_SIGNED, &convert_imgs_[i], strides);
        }
    }

    //内存分配
    bm_image_alloc_contiguous_mem(max_infer_buffer_, infer_imgs_.data());
    bm_image_alloc_contiguous_mem(max_batch_, resize_imgs_.data());
    bm_image_alloc_contiguous_mem(max_batch_, convert_imgs_.data());

    //连续的image内存中获取连续的device内存
    bm_image_get_contiguous_device_mem(max_infer_buffer_, infer_imgs_.data(), &infer_device_mem_);
    bm_image_get_contiguous_device_mem(max_batch_, resize_imgs_.data(), &resize_device_mem_);
    bm_image_get_contiguous_device_mem(max_batch_, convert_imgs_.data(), &convert_device_mem_);

    //设定推理时输入数据的device内存地址
    inputs[0]->set_device_mem(convert_device_mem_);

    //将device内存映射出来 得到虚拟地址
    bm_mem_mmap_device_mem(algo_handle_, &resize_device_mem_, &resize_buff_addr_);
    bm_mem_mmap_device_mem(algo_handle_, &infer_device_mem_, &infer_buff_addr_);

   process_status = bm_mem_mmap_device_mem(algo_handle_, (bm_device_mem_t*)&(output_tensor_->device_mem), &out_buff_addr_);
   assert(BM_SUCCESS == process_status)
   algo_init_state_ = true;
   return 0;
}

int InferModel::Preprocess(cv::Mat &input_data, int data_idx)
{
    cv::bmcv::toBMI((cv::Mat&)input_data, &infer_imgs_[data_idx]);
    auto ret = bmcv_image_vpp_convert(algo_handle_, 1, infer_imgs_[data_idx], &resize_imgs_[data_idx]);
    ret = bmcv_image_convert_to(algo_handle_, 1, preprocess_attr_, &resize_imgs_[data_idx], &convert_imgs_[data_idx]);
    return 0;
}

// int InferModel::PreprocessInModel(int batch_idx)
// {
//     int start_pos = (batch_idx << 2) * single_img_length_; //max_batch_=4  <<2
//     bm_mem_flush_partial_device_mem(algo_handle_, &infer_device_mem_, start_pos, single_img_length_ << 2);
//     autp &inputs = model_net_->Inputs();
//     bm_device_mem_t device_infer;
//     bm_image_get_contiguous_device_mem(max_batch_, &infer_imgs_[batch_idx << 2], &device_infer);
//     inputs[0]->set_device_mem(device_infer);
//     return 0;
// }

int InferModel::Inference()
{
    auto ret = model_net_->Forward();
    CV_Assert(BM_SUCCESS == ret);
    bm_mem_invalidate_device_mem(algo_handle_, (bm_device_mem_t*)&(output_tensor_->device_mem));
    return 0;
}