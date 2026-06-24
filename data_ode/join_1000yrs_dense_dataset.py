
import os
import numpy as np

def join_files(base_file_path, output_file_path):
    part_num = 1
    with open(output_file_path, 'wb') as dest_file:
        while True:
            part_filename = f"{base_file_path}.part{part_num}"
            if not os.path.exists(part_filename):
                break
            
            with open(part_filename, 'rb') as src_file:
                dest_file.write(src_file.read())
            
            print(f"joined: {part_filename}")
            part_num += 1
            
    print(f"File restoration complete: {output_file_path}")

    # verification
    try:
        data = np.load(output_file_path)
        print("Verification Success: .npz file is loaded normally with no damage.")
        print("List of arrays included:", list(data.keys()))
    except Exception as e:
        print("Verification Failed: Error occured while joining the files.", e)

if __name__=='__main__':
    # Join split files into the original 1000 years dataset of dt=0.0005
    join_files("q.npz", "q.npz")
    join_files("p.npz", "p.npz")
    join_files("dq.npz", "dq.npz")
    join_files("dp.npz", "dp.npz")
   
