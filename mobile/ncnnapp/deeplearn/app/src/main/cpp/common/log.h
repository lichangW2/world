//
// Created by clausewang(王立昌) on 2019-11-13.
//

#ifndef DEEPLEARN_LOG_H
#define DEEPLEARN_LOG_H

#include <android/log.h>

static const char* MOBILE_CNN = "Mobile-Dnn-runner";
#define LOGI(...) \
  ((void)__android_log_print(ANDROID_LOG_INFO, MOBILE_CNN, __VA_ARGS__))
#define LOGW(...) \
  ((void)__android_log_print(ANDROID_LOG_WARN, MOBILE_CNN, __VA_ARGS__))
#define LOGE(...) \
  ((void)__android_log_print(ANDROID_LOG_ERROR, MOBILE_CNN, __VA_ARGS__))

#endif //DEEPLEARN_LOG_H
