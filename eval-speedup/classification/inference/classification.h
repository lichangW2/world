#ifndef CLASSIFICATION_H
#define CLASSIFICATION_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

typedef struct classifier_ctx classifier_ctx;

classifier_ctx* classifier_initialize(const char* name, const char* model_file, const char* trained_file,
                                      int instance_num);

const char* classifier_classify(classifier_ctx* ctx,
                                const char* buffer, size_t length);

void classifier_destroy(classifier_ctx* ctx);

#ifdef __cplusplus
}
#endif

#endif // CLASSIFICATION_H

