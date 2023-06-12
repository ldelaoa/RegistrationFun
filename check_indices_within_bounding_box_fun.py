def is_inside_bounding_box(tumor_index, bounding_box):
    # Extracting bounding box coordinates
    min_col, min_row, min_slice, max_col, max_row, max_slice = bounding_box

    col, row, slice_ = tumor_index
    
    return (
        min_col <= col <= max_col and
        min_row <= row <= max_row and
        min_slice <= slice_ <= max_slice
    )

def check_indices_within_bounding_box(image, bounding_box):
    tumor_indices = np.where(image == 1)
    for i in range(len(tumor_indices[0])):
        index = [tumor_indices[0][i],tumor_indices[1][i],tumor_indices[2][i]]
        if not is_inside_bounding_box(index, bounding_box):
            print("Coord out of bb: ", index, bounding_box)
            return False,index
    return True,0
