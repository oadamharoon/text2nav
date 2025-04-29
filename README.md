# text2nav

## Setup installation for Isaac Sim on Windows 
If you encounter the following error 
`[Error] [carb.graphics-vulkan.plugin] VK_EXT_memory_budget is not supported on this platform. Can't get the VRAM memory usage info
 [Warning] [gpu.foundation.plugin] ResourceAllocator: Can't get memory usage info. All unused memory will be released`
We can fix this by removing the `OpenCL™, OpenGL®, and Vulkan® Compatibility Pack` from Add and Remove Programs in Windows Settings. https://forums.developer.nvidia.com/t/memory-errors-in-isaac-sim-and-lab-on-windows-11-with-rtx-4070/326089/4
