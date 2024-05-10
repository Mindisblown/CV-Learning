//
// Created by wsl on 2021/4/9.
// Contact: 2501038982@qq.com
//

#ifndef INFER_OCR_INT8ENTROPYCALIBRATOR_H
#define INFER_OCR_INT8ENTROPYCALIBRATOR_H
#include <cudnn.h>
#include <string>
#include <vector>
#include "NvInfer.h"
#include "utils.h"

class Int8EntropyCalibrator : public nvinfer1::IInt8EntropyCalibrator2 {
public:
	Int8EntropyCalibrator(int BatchSize, const std::vector<std::vector<float>>& data,
		const std::string& CalibDataName = "", bool readCache = true);

	virtual ~Int8EntropyCalibrator();

	int getBatchSize() const noexcept override {
		std::cout << "getbatchSize: " << mBatchSize << std::endl;
		return mBatchSize;
	}

	bool getBatch(void* bindings[], const char* names[], int nbBindings) noexcept override;

	const void* readCalibrationCache(size_t& length) noexcept override;

	void writeCalibrationCache(const void* cache, size_t length) noexcept override;

private:
	std::string mCalibDataName;
	std::vector<std::vector<float>> mDatas;
	int mBatchSize;

	int mCurBatchIdx;
	float* mCurBatchData{ nullptr };

	size_t mInputCount;
	bool mReadCache;
	void* mDeviceInput{ nullptr };

	std::vector<char> mCalibrationCache;
};
#endif //INFER_OCR_INT8ENTROPYCALIBRATOR_H
