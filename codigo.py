def process_file(file_data):
    with open("processed_file.txt", "wb") as f:
        f.write(file_data)