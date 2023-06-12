def CropBinary(binary_image,Middle_bool=False,coords_extra=None,extra=5):

    slices = np.any(binary_image, axis=(0, 1))
    rows = np.any(binary_image, axis=(0, 2))
    cols = np.any(binary_image, axis=(1, 2))
    
    min_slice, max_slice = np.where(slices)[0][[0, -1]]
    min_row, max_row = np.where(rows)[0][[0, -1]]
    min_col, max_col = np.where(cols)[0][[0, -1]]
    min_slice =min_slice - extra if min_slice > extra else 0
    min_col =min_col- extra if min_col > extra else 0
    min_row =min_row- extra if min_row > extra else 0
    
    max_row = max_row+ extra
    max_slice = max_slice+ (extra//2)
    max_col =max_col+ extra 

    cropped_image = binary_image[min_slice:max_slice+1, min_row:max_row+1, min_col:max_col+1]
    
    return cropped_image,[min_col,min_row,min_slice,max_col,max_row,max_slice]
                        
