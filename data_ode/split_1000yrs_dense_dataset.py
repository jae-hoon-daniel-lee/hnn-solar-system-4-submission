
import os

def split_file(file_path, chunk_size_mb=50):
    chunk_size = chunk_size_mb * 1024 * 1024  # MB to Bytes
    file_size = os.path.getsize(file_path)
    print(f"Original file szie: {file_size / (1024*1024):.2f} MB")

    with open(file_path, 'rb') as src_file:
        part_num = 1
        while True:
            chunk = src_file.read(chunk_size)
            if not chunk:
                break
            
            part_filename = f"{file_path}.part{part_num}"
            with open(part_filename, 'wb') as dest_file:
                dest_file.write(chunk)
            
            print(f"created: {part_filename} ({len(chunk) / (1024*1024):.2f} MB)")
            part_num += 1

if __name__=='__main__':
    # split 1000 years dataset generated in dt=0.0005 into smaller pieces
    split_file("q.npz", chunk_size_mb=50)
    split_file("p.npz", chunk_size_mb=50)
    split_file("dq.npz", chunk_size_mb=50)
    split_file("dp.npz", chunk_size_mb=50)