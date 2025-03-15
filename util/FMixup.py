import numpy as np

def amp_spectrum_blend( amp_local, amp_auxiliary, alpha=0.1 , ratio=0):
    
    a_local = np.fft.fftshift( amp_local, axes=(-2, -1) )
    a_auxiliary = np.fft.fftshift( amp_auxiliary, axes=(-2, -1) )

    _, h, w = a_local.shape
    b = (  np.floor(np.amin((h,w))*alpha)  ).astype(int)
    c_h = np.floor(h/2.0).astype(int)
    c_w = np.floor(w/2.0).astype(int)

    h1 = c_h-b
    h2 = c_h+b+1
    w1 = c_w-b
    w2 = c_w+b+1

    a_local[:,h1:h2,w1:w2] = a_local[:,h1:h2,w1:w2] * ratio + a_auxiliary[:,h1:h2,w1:w2] * (1- ratio)
    a_local = np.fft.ifftshift( a_local, axes=(-2, -1) )
    return a_local

def freq_mixup_interpolation(local_time_series, auxiliary_time_series, alpha=0 , ratio=0):
    local_time_series_np = local_time_series.cpu().detach().numpy() 
    fft_local_np = np.fft.fft2(local_time_series_np, axes=(-2, -1))
    amp_local, pha_local = np.abs(fft_local_np), np.angle(fft_local_np)

    auxiliary_time_series_np = auxiliary_time_series.cpu().detach().numpy() 
    fft_auxiliary_np = np.fft.fft2(auxiliary_time_series_np, axes=(-2, -1))

    amp_auxiliary = np.abs(fft_auxiliary_np)

    amp_local_ = amp_spectrum_blend( amp_local, amp_auxiliary, alpha=alpha , ratio=ratio)

    fft_local_ = amp_local_ * np.exp( 1j * pha_local )
    mixup_local = np.fft.ifft2( fft_local_, axes=(-2, -1) )
    mixup_local = np.real(mixup_local)

    return mixup_local