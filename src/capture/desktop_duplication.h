#pragma once

/**
 * @file desktop_duplication.h
 * @brief Educational implementation of Desktop Duplication API capture
 * 
 * This file demonstrates GPU-based screen capture using Windows Desktop
 * Duplication API. This is a LEGAL and NON-INVASIVE method of capturing
 * screen content that works at the GPU level without injecting into processes.
 * 
 * @warning FOR EDUCATIONAL PURPOSES ONLY
 * @author Game Vision Security Research Project
 */

#include <d3d11.h>
#include <dxgi1_2.h>
#include <wrl/client.h>
#include <memory>
#include <vector>

using Microsoft::WRL::ComPtr;

namespace gvsr {
namespace capture {

/**
 * @brief Result codes for capture operations
 */
enum class CaptureResult {
    Success,
    NotInitialized,
    DeviceRemoved,
    AccessLost,
    Timeout,
    UnexpectedError
};

/**
 * @brief Frame data structure
 */
struct CapturedFrame {
    std::vector<uint8_t> data;  ///< BGRA pixel data
    uint32_t width;              ///< Frame width
    uint32_t height;             ///< Frame height
    uint32_t pitch;              ///< Row pitch in bytes
    uint64_t timestamp_us;       ///< Capture timestamp in microseconds
};

/**
 * @class DesktopDuplication
 * @brief GPU-based screen capture using Desktop Duplication API
 * 
 * This class provides a clean interface to capture screen content using
 * the official Windows Desktop Duplication API (DXGI). This method:
 * - Does NOT inject into game processes
 * - Does NOT modify game memory
 * - Works at GPU driver level
 * - Is completely legal and ethical for research
 * 
 * Example usage:
 * @code
 * DesktopDuplication capture;
 * if (capture.Initialize(0)) { // Monitor 0
 *     CapturedFrame frame;
 *     if (capture.CaptureFrame(frame) == CaptureResult::Success) {
 *         // Process frame data
 *     }
 * }
 * @endcode
 */
class DesktopDuplication {
public:
    DesktopDuplication();
    ~DesktopDuplication();

    // Prevent copying
    DesktopDuplication(const DesktopDuplication&) = delete;
    DesktopDuplication& operator=(const DesktopDuplication&) = delete;

    /**
     * @brief Initialize the capture system for a specific monitor
     * @param adapter_index GPU adapter index (usually 0)
     * @param output_index Monitor index (0 = primary monitor)
     * @return true if successful
     */
    bool Initialize(uint32_t adapter_index = 0, uint32_t output_index = 0);

    /**
     * @brief Capture a single frame from the desktop
     * @param frame Output frame data
     * @param timeout_ms Maximum time to wait for a frame (0 = non-blocking)
     * @return CaptureResult indicating success or failure reason
     */
    CaptureResult CaptureFrame(CapturedFrame& frame, uint32_t timeout_ms = 100);

    /**
     * @brief Release the current frame (must be called after each capture)
     */
    void ReleaseFrame();

    /**
     * @brief Check if the capture system is initialized
     * @return true if ready to capture
     */
    bool IsInitialized() const { return m_initialized; }

    /**
     * @brief Get the desktop dimensions
     * @param width Output width
     * @param height Output height
     * @return true if initialized
     */
    bool GetDesktopDimensions(uint32_t& width, uint32_t& height) const;

    /**
     * @brief Shutdown and release all resources
     */
    void Shutdown();

private:
    /**
     * @brief Set up D3D11 device and DXGI output
     */
    bool SetupD3D11(uint32_t adapter_index, uint32_t output_index);

    /**
     * @brief Initialize Desktop Duplication interface
     */
    bool SetupDuplication();

    /**
     * @brief Copy texture data to CPU-accessible memory
     */
    bool CopyTextureToCPU(ID3D11Texture2D* texture, CapturedFrame& frame);

    // D3D11 resources
    ComPtr<ID3D11Device> m_device;
    ComPtr<ID3D11DeviceContext> m_context;
    ComPtr<IDXGIOutput1> m_output;
    ComPtr<IDXGIOutputDuplication> m_duplication;
    
    // Desktop info
    DXGI_OUTDUPL_DESC m_dupl_desc{};
    
    // State
    bool m_initialized = false;
    bool m_frame_acquired = false;
};

} // namespace capture
} // namespace gvsr
