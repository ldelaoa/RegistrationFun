import os

def delete_files_with_word(directory, word):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if word in file:
                file_path = os.path.join(root, file)
                os.remove(file_path)
                print(f"Deleted: {file_path}")
    return 0