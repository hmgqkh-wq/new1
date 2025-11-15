#pragma once
#include <vulkan/vulkan.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int bc_init(VkInstance instance);
int bc_register_compressed_image(VkDevice device, VkImage appImage, VkFormat format, uint32_t width, uint32_t height);
int bc_force_decompress(VkDevice device, VkImage appImage);
int bc_schedule_decompress_async(VkDevice device, VkImage appImage, VkQueue targetQueue);
void bc_shutdown(void);

#ifdef __cplusplus
}
#endif
