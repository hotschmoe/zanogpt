const std = @import("std");

// ── Handle types (opaque pointers) ──

pub const ze_driver_handle_t = *opaque {};
pub const ze_device_handle_t = *opaque {};
pub const ze_context_handle_t = *opaque {};
pub const ze_command_queue_handle_t = *opaque {};
pub const ze_command_list_handle_t = *opaque {};
pub const ze_fence_handle_t = *opaque {};
pub const ze_graph_handle_t = *opaque {};
pub const ze_event_handle_t = *opaque {};
pub const ze_graph_build_log_handle_t = *opaque {};

// ── Enums ──

pub const ze_result_t = enum(u32) {
    SUCCESS = 0,
    NOT_READY = 0x00000001,
    ERROR_DEVICE_LOST = 0x70000001,
    ERROR_OUT_OF_HOST_MEMORY = 0x70000002,
    ERROR_OUT_OF_DEVICE_MEMORY = 0x70000003,
    ERROR_UNINITIALIZED = 0x78000001,
    ERROR_UNSUPPORTED_VERSION = 0x78000002,
    ERROR_UNSUPPORTED_FEATURE = 0x78000003,
    ERROR_INVALID_ARGUMENT = 0x78000004,
    ERROR_INVALID_NULL_HANDLE = 0x78000005,
    ERROR_INVALID_NULL_POINTER = 0x78000006,
    ERROR_INVALID_ENUMERATION = 0x7800000D,
    _,
};

pub const ze_device_type_t = enum(u32) {
    GPU = 1,
    CPU = 2,
    FPGA = 3,
    MCA = 4,
    VPU = 5,
    _,
};

pub const ze_structure_type_t = enum(u32) {
    DEVICE_PROPERTIES = 0x00010002,
    CONTEXT_DESC = 0x0002000E,
    COMMAND_QUEUE_DESC = 0x00030001,
    COMMAND_LIST_DESC = 0x00030002,
    FENCE_DESC = 0x00030003,
    DEVICE_MEM_ALLOC_DESC = 0x00040001,
    HOST_MEM_ALLOC_DESC = 0x00040002,
    _,
};

pub const ze_command_queue_mode_t = enum(u32) {
    DEFAULT = 0,
    SYNCHRONOUS = 1,
    ASYNCHRONOUS = 2,
    _,
};

pub const ze_command_queue_priority_t = enum(u32) {
    NORMAL = 0,
    PRIORITY_LOW = 1,
    PRIORITY_HIGH = 2,
    _,
};

pub const ze_graph_format_t = enum(u32) {
    NATIVE = 0x1,
    NGRAPH_LITE = 0x2,
    _,
};

// ── Constants ──

pub const ZE_INIT_FLAG_VPU_ONLY: u32 = 1 << 1;
pub const ZE_MAX_DEVICE_NAME = 256;
pub const ZE_MAX_EXTENSION_NAME = 256;
pub const MAX_U64: u64 = std.math.maxInt(u64);

// ── Extern structs (C ABI layout) ──

pub const ze_device_properties_t = extern struct {
    stype: ze_structure_type_t = .DEVICE_PROPERTIES,
    pNext: ?*anyopaque = null,
    type: ze_device_type_t = @enumFromInt(0),
    vendorId: u32 = 0,
    deviceId: u32 = 0,
    flags: u32 = 0,
    subdeviceId: u32 = 0,
    coreClockRate: u32 = 0,
    maxMemAllocSize: u64 = 0,
    maxHardwareContexts: u32 = 0,
    maxCommandQueuePriority: u32 = 0,
    numThreadsPerEU: u32 = 0,
    physicalEUSimdWidth: u32 = 0,
    numEUsPerSubslice: u32 = 0,
    numSubslicesPerSlice: u32 = 0,
    numSlices: u32 = 0,
    timerResolution: u64 = 0,
    timestampValidBits: u32 = 0,
    kernelTimestampValidBits: u32 = 0,
    uuid: [16]u8 = .{0} ** 16,
    name: [ZE_MAX_DEVICE_NAME]u8 = .{0} ** ZE_MAX_DEVICE_NAME,
};

pub const ze_context_desc_t = extern struct {
    stype: ze_structure_type_t = .CONTEXT_DESC,
    pNext: ?*anyopaque = null,
    flags: u32 = 0,
};

pub const ze_command_queue_desc_t = extern struct {
    stype: ze_structure_type_t = .COMMAND_QUEUE_DESC,
    pNext: ?*anyopaque = null,
    ordinal: u32 = 0,
    index: u32 = 0,
    flags: u32 = 0,
    mode: ze_command_queue_mode_t = .SYNCHRONOUS,
    priority: ze_command_queue_priority_t = .NORMAL,
};

pub const ze_command_list_desc_t = extern struct {
    stype: ze_structure_type_t = .COMMAND_LIST_DESC,
    pNext: ?*anyopaque = null,
    commandQueueGroupOrdinal: u32 = 0,
    flags: u32 = 0,
};

pub const ze_fence_desc_t = extern struct {
    stype: ze_structure_type_t = .FENCE_DESC,
    pNext: ?*anyopaque = null,
    flags: u32 = 0,
};

pub const ze_device_mem_alloc_desc_t = extern struct {
    stype: ze_structure_type_t = .DEVICE_MEM_ALLOC_DESC,
    pNext: ?*anyopaque = null,
    flags: u32 = 0,
    ordinal: u32 = 0,
};

pub const ze_host_mem_alloc_desc_t = extern struct {
    stype: ze_structure_type_t = .HOST_MEM_ALLOC_DESC,
    pNext: ?*anyopaque = null,
    flags: u32 = 0,
};

pub const ze_driver_extension_properties_t = extern struct {
    name: [ZE_MAX_EXTENSION_NAME]u8 = .{0} ** ZE_MAX_EXTENSION_NAME,
    version: u32 = 0,
};

/// NPU driver meta-extension DDI table (from ze_driver_npu_ext.h).
/// Contains a single pfnGetExtension for versioned extension acquisition.
pub const ze_driver_npu_dditable_ext_t = extern struct {
    pfnGetExtension: *const fn (ze_driver_handle_t, *ze_driver_extension_npu_ext_t) callconv(.c) ze_result_t,
};

/// Request struct for pfnGetExtension — passes extension name + version,
/// receives the DDI table pointer back via ppFunctionAddress.
pub const ze_driver_extension_npu_ext_t = extern struct {
    stype: u32 = 0x1, // ZE_STRUCTURE_TYPE_DRIVER_EXTENSION_NPU_EXT
    pNext: ?*anyopaque = null,
    name: [*:0]const u8,
    version: u32,
    ppFunctionAddress: *?*anyopaque,
};

pub const ze_graph_compiler_version_info_t = extern struct {
    major: u16 = 0,
    minor: u16 = 0,
};

pub const ze_device_graph_properties_t = extern struct {
    stype: u32 = 0x1, // ZE_STRUCTURE_TYPE_DEVICE_GRAPH_PROPERTIES
    pNext: ?*anyopaque = null,
    graphExtensionVersion: u32 = 0,
    compilerVersion: ze_graph_compiler_version_info_t = .{},
    graphFormatsSupported: u32 = 0,
    maxOVOpsetVersionSupported: u32 = 0,
};

pub const ze_graph_desc_t = extern struct {
    stype: u32 = 0x2, // ZE_STRUCTURE_TYPE_GRAPH_DESC
    pNext: ?*anyopaque = null,
    format: ze_graph_format_t = .NGRAPH_LITE,
    inputSize: usize = 0,
    pInput: ?[*]const u8 = null,
    pBuildFlags: ?[*:0]const u8 = null,
};

/// Version 1.5 graph descriptor — adds flags field.
pub const ze_graph_desc_2_t = extern struct {
    stype: u32 = 0xE, // ZE_STRUCTURE_TYPE_GRAPH_DESC_2
    pNext: ?*anyopaque = null,
    format: ze_graph_format_t = .NGRAPH_LITE,
    inputSize: usize = 0,
    pInput: ?[*]const u8 = null,
    pBuildFlags: ?[*:0]const u8 = null,
    flags: u32 = 0, // ZE_GRAPH_FLAG_NONE
};

pub const ze_graph_properties_t = extern struct {
    stype: u32 = 0x3,
    pNext: ?*anyopaque = null,
    numGraphArgs: u32 = 0,
};

// ── Graph DDI table (Level-Zero graph extension v1.0) ──

/// Graph extension DDI table. v1.0 fields (indices 0-8) plus stubs up to v1.5 pfnCreate2 (index 16).
pub const ze_graph_dditable_t = extern struct {
    // ── Version 1.0 (9 function pointers) ──
    pfnCreate: *const fn (ze_context_handle_t, ze_device_handle_t, *const ze_graph_desc_t, *ze_graph_handle_t) callconv(.c) ze_result_t,
    pfnDestroy: *const fn (ze_graph_handle_t) callconv(.c) ze_result_t,
    pfnGetProperties: *const fn (ze_graph_handle_t, *ze_graph_properties_t) callconv(.c) ze_result_t,
    pfnGetArgumentProperties: *const fn (ze_graph_handle_t, u32, *anyopaque) callconv(.c) ze_result_t,
    pfnSetArgumentValue: *const fn (ze_graph_handle_t, u32, *anyopaque) callconv(.c) ze_result_t,
    pfnAppendGraphInitialize: *const fn (ze_command_list_handle_t, ze_graph_handle_t, ?ze_event_handle_t, u32, ?[*]ze_event_handle_t) callconv(.c) ze_result_t,
    pfnAppendGraphExecute: *const fn (ze_command_list_handle_t, ze_graph_handle_t, ?*anyopaque, ?ze_event_handle_t, u32, ?[*]ze_event_handle_t) callconv(.c) ze_result_t,
    pfnGetNativeBinary: *const fn (ze_graph_handle_t, *usize, ?[*]u8) callconv(.c) ze_result_t,
    pfnDeviceGetGraphProperties: *const fn (ze_device_handle_t, *ze_device_graph_properties_t) callconv(.c) ze_result_t,
    // ── Version 1.1 (2 function pointers) ──
    _reserved_v1_1a: ?*anyopaque = null,
    _reserved_v1_1b: ?*anyopaque = null,
    // ── Version 1.2 (1 function pointer) ──
    _reserved_v1_2: ?*anyopaque = null,
    // ── Version 1.3 (3 function pointers) ──
    _reserved_v1_3a: ?*anyopaque = null,
    _reserved_v1_3b: ?*anyopaque = null,
    _reserved_v1_3c: ?*anyopaque = null,
    // ── Version 1.4 (1 function pointer) ──
    pfnBuildLogGetString: ?*const fn (ze_graph_handle_t, *u32, ?[*]u8) callconv(.c) ze_result_t = null,
    // ── Version 1.5 (3 function pointers) ──
    pfnCreate2: ?*const fn (ze_context_handle_t, ze_device_handle_t, *const ze_graph_desc_2_t, *ze_graph_handle_t) callconv(.c) ze_result_t = null,
    _reserved_v1_5b: ?*anyopaque = null,
    _reserved_v1_5c: ?*anyopaque = null,
    // ── Version 1.6 (1 function pointer) ──
    _reserved_v1_6: ?*anyopaque = null,
    // ── Version 1.7 (1 function pointer) ──
    _reserved_v1_7: ?*anyopaque = null,
    // ── Version 1.8 (2 function pointers) ──
    _reserved_v1_8a: ?*anyopaque = null,
    _reserved_v1_8b: ?*anyopaque = null,
    // ── Version 1.11 (2 function pointers) ──
    _reserved_v1_11a: ?*anyopaque = null,
    _reserved_v1_11b: ?*anyopaque = null,
    // ── Version 1.12 (4 function pointers) ──
    pfnCreate3: ?*const fn (ze_context_handle_t, ze_device_handle_t, *const ze_graph_desc_2_t, *ze_graph_handle_t, *?ze_graph_build_log_handle_t) callconv(.c) ze_result_t = null,
    _reserved_v1_12b: ?*anyopaque = null,
    pfnBuildLogGetString2: ?*const fn (ze_graph_build_log_handle_t, *u32, ?[*]u8) callconv(.c) ze_result_t = null,
    pfnBuildLogDestroy: ?*const fn (ze_graph_build_log_handle_t) callconv(.c) ze_result_t = null,
};

// ── Function pointer types ──

pub const pfnInit = *const fn (u32) callconv(.c) ze_result_t;
pub const pfnDriverGet = *const fn (*u32, ?[*]ze_driver_handle_t) callconv(.c) ze_result_t;
pub const pfnDeviceGet = *const fn (ze_driver_handle_t, *u32, ?[*]ze_device_handle_t) callconv(.c) ze_result_t;
pub const pfnDeviceGetProperties = *const fn (ze_device_handle_t, *ze_device_properties_t) callconv(.c) ze_result_t;
pub const pfnContextCreate = *const fn (ze_driver_handle_t, *const ze_context_desc_t, *ze_context_handle_t) callconv(.c) ze_result_t;
pub const pfnContextDestroy = *const fn (ze_context_handle_t) callconv(.c) ze_result_t;
pub const pfnCommandQueueCreate = *const fn (ze_context_handle_t, ze_device_handle_t, *const ze_command_queue_desc_t, *ze_command_queue_handle_t) callconv(.c) ze_result_t;
pub const pfnCommandQueueDestroy = *const fn (ze_command_queue_handle_t) callconv(.c) ze_result_t;

pub const pfnCommandListCreate = *const fn (ze_context_handle_t, ze_device_handle_t, *const ze_command_list_desc_t, *ze_command_list_handle_t) callconv(.c) ze_result_t;
pub const pfnCommandListDestroy = *const fn (ze_command_list_handle_t) callconv(.c) ze_result_t;
pub const pfnCommandListClose = *const fn (ze_command_list_handle_t) callconv(.c) ze_result_t;
pub const pfnCommandListReset = *const fn (ze_command_list_handle_t) callconv(.c) ze_result_t;

pub const pfnCommandQueueExecuteCommandLists = *const fn (ze_command_queue_handle_t, u32, *const ze_command_list_handle_t, ?ze_fence_handle_t) callconv(.c) ze_result_t;

pub const pfnFenceCreate = *const fn (ze_command_queue_handle_t, *const ze_fence_desc_t, *ze_fence_handle_t) callconv(.c) ze_result_t;
pub const pfnFenceDestroy = *const fn (ze_fence_handle_t) callconv(.c) ze_result_t;
pub const pfnFenceHostSynchronize = *const fn (ze_fence_handle_t, u64) callconv(.c) ze_result_t;
pub const pfnFenceReset = *const fn (ze_fence_handle_t) callconv(.c) ze_result_t;

pub const pfnMemAllocShared = *const fn (ze_context_handle_t, *const ze_device_mem_alloc_desc_t, *const ze_host_mem_alloc_desc_t, usize, usize, ze_device_handle_t, *?*anyopaque) callconv(.c) ze_result_t;
pub const pfnMemFree = *const fn (ze_context_handle_t, *anyopaque) callconv(.c) ze_result_t;

pub const pfnDriverGetExtensionFunctionAddress = *const fn (ze_driver_handle_t, [*:0]const u8, *?*anyopaque) callconv(.c) ze_result_t;
pub const pfnDriverGetExtensionProperties = *const fn (ze_driver_handle_t, *u32, ?[*]ze_driver_extension_properties_t) callconv(.c) ze_result_t;

// ── Dispatch table ──

pub const Dispatch = struct {
    zeInit: pfnInit,
    zeDriverGet: pfnDriverGet,
    zeDeviceGet: pfnDeviceGet,
    zeDeviceGetProperties: pfnDeviceGetProperties,
    zeContextCreate: pfnContextCreate,
    zeContextDestroy: pfnContextDestroy,
    zeCommandQueueCreate: pfnCommandQueueCreate,
    zeCommandQueueDestroy: pfnCommandQueueDestroy,
    zeCommandListCreate: pfnCommandListCreate,
    zeCommandListDestroy: pfnCommandListDestroy,
    zeCommandListClose: pfnCommandListClose,
    zeCommandListReset: pfnCommandListReset,
    zeCommandQueueExecuteCommandLists: pfnCommandQueueExecuteCommandLists,
    zeFenceCreate: pfnFenceCreate,
    zeFenceDestroy: pfnFenceDestroy,
    zeFenceHostSynchronize: pfnFenceHostSynchronize,
    zeFenceReset: pfnFenceReset,
    zeMemAllocShared: pfnMemAllocShared,
    zeMemFree: pfnMemFree,
    zeDriverGetExtensionFunctionAddress: pfnDriverGetExtensionFunctionAddress,
    zeDriverGetExtensionProperties: pfnDriverGetExtensionProperties,

    pub fn load(l: *std.DynLib) !Dispatch {
        var self: Dispatch = undefined;
        inline for (@typeInfo(Dispatch).@"struct".fields) |f| {
            @field(self, f.name) = l.lookup(f.type, f.name) orelse
                return error.SymbolNotFound;
        }
        return self;
    }
};

// ── Error helper ──

pub const ZeError = error{
    DeviceLost,
    OutOfHostMemory,
    OutOfDeviceMemory,
    Uninitialized,
    UnsupportedVersion,
    UnsupportedFeature,
    InvalidArgument,
    InvalidNullHandle,
    InvalidNullPointer,
    InvalidEnumeration,
    NotReady,
    Unknown,
};

pub fn check(result: ze_result_t) ZeError!void {
    return switch (result) {
        .SUCCESS => {},
        .NOT_READY => error.NotReady,
        .ERROR_DEVICE_LOST => error.DeviceLost,
        .ERROR_OUT_OF_HOST_MEMORY => error.OutOfHostMemory,
        .ERROR_OUT_OF_DEVICE_MEMORY => error.OutOfDeviceMemory,
        .ERROR_UNINITIALIZED => error.Uninitialized,
        .ERROR_UNSUPPORTED_VERSION => error.UnsupportedVersion,
        .ERROR_UNSUPPORTED_FEATURE => error.UnsupportedFeature,
        .ERROR_INVALID_ARGUMENT => error.InvalidArgument,
        .ERROR_INVALID_NULL_HANDLE => error.InvalidNullHandle,
        .ERROR_INVALID_NULL_POINTER => error.InvalidNullPointer,
        .ERROR_INVALID_ENUMERATION => error.InvalidEnumeration,
        _ => error.Unknown,
    };
}
