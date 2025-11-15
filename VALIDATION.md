# Validation instructions

Populate tests/vectors with authoritative BC6H/BC7 encoded blocks and expected decoded outputs.
Run CI or local build with NDK + shaderc to embed SPIR-V. Then run a Vulkan test harness on-device to compare outputs.
