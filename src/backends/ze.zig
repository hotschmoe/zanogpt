const std = @import("std");

// ── Handle types (opaque pointers) ──

pub const ze_driver_handle_t = *opaque {};
pub const ze_device_handle_t = *opaque {};
pub const ze_context_handle_t = *opaque {};
pub const ze_command_queue_handle_t = *opaque {};

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

// ── Constants ──

pub const ZE_INIT_FLAG_VPU_ONLY: u32 = 1 << 1;
pub const ZE_MAX_DEVICE_NAME = 256;

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

// ── Function pointer types ──

pub const pfnInit = *const fn (u32) callconv(.c) ze_result_t;
pub const pfnDriverGet = *const fn (*u32, ?[*]ze_driver_handle_t) callconv(.c) ze_result_t;
pub const pfnDeviceGet = *const fn (ze_driver_handle_t, *u32, ?[*]ze_device_handle_t) callconv(.c) ze_result_t;
pub const pfnDeviceGetProperties = *const fn (ze_device_handle_t, *ze_device_properties_t) callconv(.c) ze_result_t;
pub const pfnContextCreate = *const fn (ze_driver_handle_t, *const ze_context_desc_t, *ze_context_handle_t) callconv(.c) ze_result_t;
pub const pfnContextDestroy = *const fn (ze_context_handle_t) callconv(.c) ze_result_t;
pub const pfnCommandQueueCreate = *const fn (ze_context_handle_t, ze_device_handle_t, *const ze_command_queue_desc_t, *ze_command_queue_handle_t) callconv(.c) ze_result_t;
pub const pfnCommandQueueDestroy = *const fn (ze_command_queue_handle_t) callconv(.c) ze_result_t;

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

    pub fn load(lib: *std.DynLib) !Dispatch {
        return .{
            .zeInit = lib.lookup(pfnInit, "zeInit") orelse return error.SymbolNotFound,
            .zeDriverGet = lib.lookup(pfnDriverGet, "zeDriverGet") orelse return error.SymbolNotFound,
            .zeDeviceGet = lib.lookup(pfnDeviceGet, "zeDeviceGet") orelse return error.SymbolNotFound,
            .zeDeviceGetProperties = lib.lookup(pfnDeviceGetProperties, "zeDeviceGetProperties") orelse return error.SymbolNotFound,
            .zeContextCreate = lib.lookup(pfnContextCreate, "zeContextCreate") orelse return error.SymbolNotFound,
            .zeContextDestroy = lib.lookup(pfnContextDestroy, "zeContextDestroy") orelse return error.SymbolNotFound,
            .zeCommandQueueCreate = lib.lookup(pfnCommandQueueCreate, "zeCommandQueueCreate") orelse return error.SymbolNotFound,
            .zeCommandQueueDestroy = lib.lookup(pfnCommandQueueDestroy, "zeCommandQueueDestroy") orelse return error.SymbolNotFound,
        };
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
