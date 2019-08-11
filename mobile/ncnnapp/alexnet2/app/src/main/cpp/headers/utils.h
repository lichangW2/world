//
// Created by clausewang(王立昌) on 2019-08-03.
//

#ifndef ALEXNET_UTILS_H
#define ALEXNET_UTILS_H

#include <android/log.h>

static const char* MOBILE_CNN = "Mobile-Dnn-runner";
#define LOGI(...) \
  ((void)__android_log_print(ANDROID_LOG_INFO, MOBILE_CNN, __VA_ARGS__))
#define LOGW(...) \
  ((void)__android_log_print(ANDROID_LOG_WARN, MOBILE_CNN, __VA_ARGS__))
#define LOGE(...) \
  ((void)__android_log_print(ANDROID_LOG_ERROR, MOBILE_CNN, __VA_ARGS__))

#endif //ALEXNET_UTILS_H
