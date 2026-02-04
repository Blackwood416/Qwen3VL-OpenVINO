using System.Runtime.InteropServices;

namespace Qwen3VL;

public enum OvStatus
{
    OK = 0,
    GENERAL_ERROR = -1,
    NOT_IMPLEMENTED = -2,
    NETWORK_NOT_LOADED = -3,
    PARAMETER_MISMATCH = -4,
    NOT_FOUND = -5,
    OUT_OF_BOUNDS = -6,
    UNEXPECTED = -7,
    REQUEST_BUSY = -8,
    RESULT_NOT_READY = -9,
    NOT_ALLOCATED = -10,
    INFER_NOT_STARTED = -11,
    NETWORK_NOT_READ = -12,
    INFER_CANCELLED = -13,
    INVALID_C_PARAM = -14,
    UNKNOWN_C_ERROR = -15,
    NOT_IMPLEMENT_C_METHOD = -16,
    UNKNOWN_EXCEPTION = -17
}

public enum OvElementType
{
    UNDEFINED = 0,
    BOOL = 1,
    // BIT = 2,
    // U8 = 3,
    F32 = 4,
    F16 = 5,
    I8 = 6,
    I16 = 7,
    I32 = 9,
    I64 = 10,
    U32 = 12,
    U64 = 13,
    BF16 = 14,
    STRING = 16
}

[StructLayout(LayoutKind.Sequential)]
public struct OvShape
{
    public nuint Rank;
    public nint Dims; // long*
}

[StructLayout(LayoutKind.Sequential)]
public struct OvDimension
{
    public long Min;
    public long Max;
}

[StructLayout(LayoutKind.Sequential)]
public struct OvPartialShape
{
    public OvDimension Rank;
    public nint Dims; // OvDimension*
}

public static class NativeMethods
{
    private const string CoreDll = "openvino_c.dll";

    [DllImport(CoreDll, CallingConvention = CallingConvention.Cdecl)]
    public static extern OvStatus ov_core_create(out nint core);

    [DllImport(CoreDll, CallingConvention = CallingConvention.Cdecl)]
    public static extern void ov_core_free(nint core);

    [DllImport(CoreDll, CallingConvention = CallingConvention.Cdecl)]
    public static extern OvStatus ov_core_read_model(nint core, [MarshalAs(UnmanagedType.LPUTF8Str)] string modelPath, [MarshalAs(UnmanagedType.LPUTF8Str)] string? binPath, out nint model);

    [DllImport(CoreDll, CallingConvention = CallingConvention.Cdecl)]
    public static extern void ov_model_free(nint model);

    [DllImport(CoreDll, CallingConvention = CallingConvention.Cdecl)]
    public static extern OvStatus ov_core_compile_model(nint core, nint model, [MarshalAs(UnmanagedType.LPUTF8Str)] string deviceName, nuint propertyArgsSize, out nint compiledModel);

    [DllImport(CoreDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "ov_core_compile_model")]
    public static extern OvStatus ov_core_compile_model_props(nint core, nint model, [MarshalAs(UnmanagedType.LPUTF8Str)] string deviceName, nuint propertyArgsSize, out nint compiledModel,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string k1, [MarshalAs(UnmanagedType.LPUTF8Str)] string v1,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string k2, [MarshalAs(UnmanagedType.LPUTF8Str)] string v2,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string k3, [MarshalAs(UnmanagedType.LPUTF8Str)] string v3);

    [DllImport(CoreDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "ov_core_compile_model")]
    public static extern OvStatus ov_core_compile_model_props5(nint core, nint model, [MarshalAs(UnmanagedType.LPUTF8Str)] string deviceName, nuint propertyArgsSize, out nint compiledModel,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string k1, [MarshalAs(UnmanagedType.LPUTF8Str)] string v1,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string k2, [MarshalAs(UnmanagedType.LPUTF8Str)] string v2,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string k3, [MarshalAs(UnmanagedType.LPUTF8Str)] string v3,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string k4, [MarshalAs(UnmanagedType.LPUTF8Str)] string v4,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string k5, [MarshalAs(UnmanagedType.LPUTF8Str)] string v5);

    [DllImport(CoreDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "ov_core_compile_model")]
    public static extern OvStatus ov_core_compile_model_props8(nint core, nint model, [MarshalAs(UnmanagedType.LPUTF8Str)] string deviceName, nuint propertyArgsSize, out nint compiledModel,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string k1, [MarshalAs(UnmanagedType.LPUTF8Str)] string v1,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string k2, [MarshalAs(UnmanagedType.LPUTF8Str)] string v2,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string k3, [MarshalAs(UnmanagedType.LPUTF8Str)] string v3,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string k4, [MarshalAs(UnmanagedType.LPUTF8Str)] string v4,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string k5, [MarshalAs(UnmanagedType.LPUTF8Str)] string v5,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string k6, [MarshalAs(UnmanagedType.LPUTF8Str)] string v6,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string k7, [MarshalAs(UnmanagedType.LPUTF8Str)] string v7,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string k8, [MarshalAs(UnmanagedType.LPUTF8Str)] string v8);

    [DllImport(CoreDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "ov_core_compile_model")]
    public static extern OvStatus ov_core_compile_model_props10(nint core, nint model, [MarshalAs(UnmanagedType.LPUTF8Str)] string deviceName, nuint propertyArgsSize, out nint compiledModel,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string k1, [MarshalAs(UnmanagedType.LPUTF8Str)] string v1,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string k2, [MarshalAs(UnmanagedType.LPUTF8Str)] string v2,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string k3, [MarshalAs(UnmanagedType.LPUTF8Str)] string v3,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string k4, [MarshalAs(UnmanagedType.LPUTF8Str)] string v4,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string k5, [MarshalAs(UnmanagedType.LPUTF8Str)] string v5,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string k6, [MarshalAs(UnmanagedType.LPUTF8Str)] string v6,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string k7, [MarshalAs(UnmanagedType.LPUTF8Str)] string v7,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string k8, [MarshalAs(UnmanagedType.LPUTF8Str)] string v8,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string k9, [MarshalAs(UnmanagedType.LPUTF8Str)] string v9,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string k10, [MarshalAs(UnmanagedType.LPUTF8Str)] string v10);

    [DllImport(CoreDll, CallingConvention = CallingConvention.Cdecl)]
    public static extern nint ov_get_last_err_msg();

    [DllImport(CoreDll, CallingConvention = CallingConvention.Cdecl)]
    public static extern void ov_compiled_model_free(nint compiledModel);

    [DllImport(CoreDll, CallingConvention = CallingConvention.Cdecl)]
    public static extern OvStatus ov_compiled_model_create_infer_request(nint compiledModel, out nint inferRequest);

    [DllImport(CoreDll, CallingConvention = CallingConvention.Cdecl)]
    public static extern OvStatus ov_compiled_model_get_property(nint compiledModel, [MarshalAs(UnmanagedType.LPUTF8Str)] string name, out nint valuePtr);

    [DllImport(CoreDll, CallingConvention = CallingConvention.Cdecl)]
    public static extern void ov_infer_request_free(nint inferRequest);

    [DllImport(CoreDll, CallingConvention = CallingConvention.Cdecl)]
    public static extern OvStatus ov_infer_request_get_tensor(nint inferRequest, [MarshalAs(UnmanagedType.LPUTF8Str)] string tensorName, out nint tensor);

    [DllImport(CoreDll, CallingConvention = CallingConvention.Cdecl)]
    public static extern OvStatus ov_infer_request_set_tensor(nint inferRequest, [MarshalAs(UnmanagedType.LPUTF8Str)] string tensorName, nint tensor);

    [DllImport(CoreDll, CallingConvention = CallingConvention.Cdecl)]
    public static extern OvStatus ov_infer_request_get_input_tensor_by_index(nint inferRequest, nuint index, out nint tensor);

    [DllImport(CoreDll, CallingConvention = CallingConvention.Cdecl)]
    public static extern OvStatus ov_infer_request_get_output_tensor_by_index(nint inferRequest, nuint index, out nint tensor);

    [DllImport(CoreDll, CallingConvention = CallingConvention.Cdecl)]
    public static extern OvStatus ov_infer_request_set_input_tensor_by_index(nint inferRequest, nuint index, nint tensor);

    [DllImport(CoreDll, CallingConvention = CallingConvention.Cdecl)]
    public static extern OvStatus ov_infer_request_infer(nint inferRequest);

    [DllImport(CoreDll, CallingConvention = CallingConvention.Cdecl)]
    public static extern OvStatus ov_infer_request_start_async(nint inferRequest);

    [DllImport(CoreDll, CallingConvention = CallingConvention.Cdecl)]
    public static extern OvStatus ov_infer_request_wait(nint inferRequest);

    [DllImport(CoreDll, CallingConvention = CallingConvention.Cdecl)]
    public static extern OvStatus ov_model_inputs_size(nint model, out nuint size);

    [DllImport(CoreDll, CallingConvention = CallingConvention.Cdecl)]
    public static extern OvStatus ov_model_outputs_size(nint model, out nuint size);

    [DllImport(CoreDll, CallingConvention = CallingConvention.Cdecl)]
    public static extern OvStatus ov_shape_create(nuint rank, long[] dims, out OvShape shape);

    [DllImport(CoreDll, CallingConvention = CallingConvention.Cdecl)]
    public static extern void ov_shape_free(ref OvShape shape);

    [DllImport(CoreDll, CallingConvention = CallingConvention.Cdecl)]
    public static extern OvStatus ov_tensor_create(OvElementType type, OvShape shape, out nint tensor);

    [DllImport(CoreDll, CallingConvention = CallingConvention.Cdecl)]
    public static extern OvStatus ov_tensor_create_from_host_ptr(OvElementType type, OvShape shape, nint hostPtr, out nint tensor);

    [DllImport(CoreDll, CallingConvention = CallingConvention.Cdecl)]
    public static extern void ov_tensor_free(nint tensor);

    [DllImport(CoreDll, CallingConvention = CallingConvention.Cdecl)]
    public static extern OvStatus ov_tensor_data(nint tensor, out nint dataPtr);

    [DllImport(CoreDll, CallingConvention = CallingConvention.Cdecl)]
    public static extern OvStatus ov_tensor_get_shape(nint tensor, out OvShape shape);

    [DllImport(CoreDll, CallingConvention = CallingConvention.Cdecl)]
    public static extern OvStatus ov_tensor_set_shape(nint tensor, OvShape shape);

    [DllImport(CoreDll, CallingConvention = CallingConvention.Cdecl)]
    public static extern OvStatus ov_tensor_get_element_type(nint tensor, out OvElementType type);

    [DllImport(CoreDll, CallingConvention = CallingConvention.Cdecl)]
    public static extern OvStatus ov_tensor_get_size(nint tensor, out nuint size);

    [DllImport(CoreDll, CallingConvention = CallingConvention.Cdecl)]
    public static extern OvStatus ov_tensor_get_byte_size(nint tensor, out nuint size);

    [DllImport(CoreDll, CallingConvention = CallingConvention.Cdecl)]
    public static extern OvStatus ov_model_input_by_index(nint model, nuint index, out nint port);

    [DllImport(CoreDll, CallingConvention = CallingConvention.Cdecl)]
    public static extern OvStatus ov_model_output_by_index(nint model, nuint index, out nint port);

    [DllImport(CoreDll, CallingConvention = CallingConvention.Cdecl)]
    public static extern OvStatus ov_port_get_any_name(nint port, out nint namePtr);

    [DllImport(CoreDll, CallingConvention = CallingConvention.Cdecl)]
    public static extern OvStatus ov_port_get_element_type(nint port, out OvElementType type);

    [DllImport(CoreDll, CallingConvention = CallingConvention.Cdecl)]
    public static extern OvStatus ov_port_get_partial_shape(nint port, out OvPartialShape shape);

    [DllImport(CoreDll, CallingConvention = CallingConvention.Cdecl)]
    public static extern void ov_free(nint content);

    // Variable State API
    [DllImport(CoreDll, CallingConvention = CallingConvention.Cdecl)]
    public static extern OvStatus ov_infer_request_query_state(nint inferRequest, out nint states, out nuint size);

    [DllImport(CoreDll, CallingConvention = CallingConvention.Cdecl)]
    public static extern void ov_variable_state_free(nint state);

    [DllImport(CoreDll, CallingConvention = CallingConvention.Cdecl)]
    public static extern OvStatus ov_variable_state_get_name(nint state, out nint namePtr);

    [DllImport(CoreDll, CallingConvention = CallingConvention.Cdecl)]
    public static extern OvStatus ov_variable_state_get_state(nint state, out nint tensor);

    [DllImport(CoreDll, CallingConvention = CallingConvention.Cdecl)]
    public static extern OvStatus ov_variable_state_set_state(nint state, nint tensor);
}
