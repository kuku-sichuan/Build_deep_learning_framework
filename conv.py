import numpy as np

def origin_conv_forward(features, kernel, bias, conv_params):
    """
    :param features: (N, C_in, H, W)
    :param kernel: (C_out, C_in, h, w)
    :param bias: (C_out)
    :param conv_params: dict of conv params;
             padding: the number of pixels to pad;
             stride: The number of pixels to override
    :return: the output of feature(N, C_out, new_H, new_W)
    """
    pad = conv_params["padding"]
    s = conv_params["stride"]

    k_cout, k_cin, k_h, k_w = np.shape(kernel)
    n, c, H, W = np.shape(features)
    features_pad = np.pad(features, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="constant")
    new_H, new_W = int((H + 2 * pad - k_h) / s + 1), int((W + 2 * pad - k_w) / s + 1)
    output = np.zeros((n, k_cout, new_H, new_W))

    for i in range(n):
        for c_out in range(k_cout):
            for h in range(new_H):
                for w in range(new_W):
                    output[i][c_out][h][w] = \
                        np.sum(features_pad[i, :, s*h:(s*h+k_h),\
                               s*w:(s*w+k_w)]*kernel[c_out]) + bias[c_out]
    cache = (features, kernel, bias, conv_params)
    return output, cache


def origin_conv_backward(dout, cache):
    """
    :param dout: Up stream derivatives (N,C,new_H, new_W).
    :param cache: A tuple of (features, kernel, bias, comv_params)
    :return: a tuple of:
       - df: Gradient with respect to features
       - dk: Gradient with respect to kernels
       - db: Gradient with respect to bias
    """
    features, kernel, bias, conv_params = cache
    s, pad = conv_params["stride"], conv_params["padding"]
    df = np.zeros_like(features)
    dk = np.zeros_like(kernel)
    db = np.zeros_like(bias)

    N,C,H,W = features.shape
    k_cout, k_cin, k_h, k_w = kernel.shape
    features_pad = np.pad(features, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="constant")
    df_pad = np.pad(df, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="constant")
    new_H, new_W = int((H+2*pad-k_h)/s + 1), int(((W+2*pad-k_w)/s + 1))
    '''
    out = kernel*f + b
    
    db = dout # N,H,W dims sum up to get db.
    dk = dout*f # dout is a pixel in c-th channel, f is (C_in, k_h, k_w) for compute out. 
                # related channel is kernel(c)---[C_in, k_h, k_w]
    df = dout*kernel # dout is a pixel in c-th channel, related channel is kernel(k_cout)---[C_in, k_h, k_w]
                     # related features is (C_in, k_h, k_w)
    '''
    for i in range(N):
        for c in range(k_cout):
            db[c] += np.sum(dout[i, c, :, :])
            for h in range(new_H):
                for w in range(new_W):
                    # windows is (C_in, k_h, k_w)
                    windows = features_pad[i, :, h*s:h*s+k_h, w*s:w*s+k_w]
                    dk[c] += dout[i, c, h, w]*windows
                    df_pad[i, :, h*s:h*s+k_h, w*s:w*s+k_w] += \
                        kernel[c]*dout[i, c, h, w]

    df = df_pad[:, :, pad:H+pad, pad:W+pad]
    return df, dk, db


def im2col(feature, k_h, k_w, stride, padding):
    """
    :param feature: (N, C_in, H, W)
    :param k_h: the height of kernel
    :param k_w: the width of kernel
    :param stride: the stride of conv
    :param padding: the number of pixels to pad
    :return:(N, new_W*new_H, C_in*k_h*k_w)

    Sketch Map:(k=3)
                    ----------c1---------|------c2-----|.....|------c_in-----|
                      h      (k_h*k_w)    w
          -------------------
         |                  |
    [|1,2,3|,4]       1[1,2,3,2,3,4,3,4,5]1                                       (C_IN*k_h*k_w, C_OUT)
    [|2,3,4|,5]------>1[2,3,4,3,4,5,4,5,6]2                                  *   [c1,c2,..., cn]^T = [new_h*new_w,C_OUT]
    [|3,4,5|,6]       2[2,3,4,3,4,5,4,5,6]1
    [4,5,6,7]         2[3,4,5,4,5,6,5,6,7]2
    """
    n, c, H, W = np.shape(feature)
    pad = padding
    s = stride
    features_pad = np.pad(feature, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="constant")
    new_H, new_W = int((H + 2 * pad - k_h) / s + 1), int((W + 2 * pad - k_w) / s + 1)
    im2col_feature = np.zeros((n, new_W*new_H, c*k_h*k_w))
    for i in range(n):
        for h in range(new_H):
            for w in range(new_W):
                im2col_feature[i, h*new_W+w, :] = \
                    np.reshape(features_pad[i, :, s*h:s*h+k_h, s*w:s*w+k_w],\
                               (c*k_h*k_w))

    return im2col_feature, new_H, new_W

def im2col_back(df, feature_shape, k_h, k_w, stride, padding):
    """
    :param df: the derivatives of features (N, new_H*new_W, k_cin*k_h*k_w)
    :param feature_shape: (N,k_cin,H,W)
    :param k_h:
    :param k_w:
    :param stride:
    :param padding:
    :return:df_out (N, k_cin, H, W)
    """
    s = stride
    pad = padding
    n, k_cin, H, W = feature_shape
    pad_H, pad_W = H + 2 * pad, W + 2 * pad
    df_out = np.zeros((n, k_cin, pad_H, pad_W))
    new_H, new_W = int((H+2*pad-k_h)/s+1), int((W+2*pad-k_w)/s+1)
    for i in range(n):
        for h in range(new_H):
            for w in range(new_W):
                # Reversing the assignment above
                df_out[i, :, s*h:s*h+k_h, s*w:s*w+k_w] += \
                    np.reshape(df[i, h*new_W+w, :], (k_cin, k_h, k_w))
    df_out = df_out[:, :, pad:pad+H, pad:pad+W]
    return df_out


def im2col_conv_forward(features, kernel, bias, conv_params):
    """
    :param features: (N, C_in, H, W)
    :param kernel: (C_out, C_in, h, w)
    :param bias: (C_out)
    :param conv_params: dict of conv params;
             padding: "same" or "valid";
             stride: The number of pixels to override
    :return: the output of feature(N, C_out, new_H, new_W)
    """
    pad = conv_params["padding"]
    s = conv_params["stride"]
    k_cout, k_cin, k_h, k_w = np.shape(kernel)
    # im2col_feature's shape is (N, new_H*new_W, C_in*k_h*k_w)
    im2col_feature, new_H, new_W = im2col(features, k_h, k_w, s, pad)
    # im2col_kernel's shape is ( C_in*k_h*k_w, C_out)
    im2col_kernel = np.transpose(np.reshape(kernel, (k_cout, k_cin*k_h*k_w)), (1,0))
    # result's shape is (N, new_H*new_W, C_out)
    result = np.dot(im2col_feature, im2col_kernel) + np.reshape(bias, (1, 1, -1))
    result = np.reshape(result, (features.shape[0], new_H, new_W, k_cout))
    result = np.transpose(result, (0, 3, 1, 2))
    cache = (features, im2col_feature, kernel, bias, conv_params)
    return result, cache


def im2col_conv_backward(dout, cache):
    """
    :param dout: Up stream derivatives (N, k_cout, new_H, new_W).
    :param cache: A tuple of (features, kernel, bias, conv_params)
    :return: a tuple of:
       - df: Gradient with respect to features
       - dk: Gradient with respect to kernels
       - db: Gradient with respect to bias
    """
    features, im2col_feature, kernel, bias, conv_params = cache
    s, pad = conv_params["stride"], conv_params["padding"]
    # im2col_feature is (N, new_W*new_H, k_cin*k_h*k_w)
    N, new_HW, Ckhw = im2col_feature.shape

    k_cout, k_cin, k_h, k_w = kernel.shape
    # dout_reshaped is (N, new_W*new_H, k_cout)
    dout_reshaped = np.transpose(np.reshape(dout, (N, k_cout, new_HW)), (0, 2, 1))
    db = np.sum(dout_reshaped, axis=(0,1))

    dk = np.zeros((k_h*k_w*k_cin, k_cout))
    for i in range(N):
        dk += np.dot(np.transpose(im2col_feature, (0,2,1))[i], dout_reshaped[i])
    dk = np.reshape(dk, (k_cin, k_h, k_w, k_cout))
    dk = np.transpose(dk, (3, 0, 1, 2))

    # kernel_reshaped is (k_cout, k_cin*k_h*k_w)
    kernel_reshaped = np.reshape(kernel, (k_cout, k_cin*k_h*k_w))
    # df is (N, new_W*new_H, k_cin*k_h*k_w)
    df = np.dot(dout_reshaped, kernel_reshaped)
    # df is (N, k_cin, H, W)
    df = im2col_back(df, features.shape, k_h, k_w, s, pad)

    return df, dk, db
