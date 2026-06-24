
import torch
import os
import sys

common_dir_name = 'commons' 
common_dir_path = os.path.join(os.path.dirname(os.getcwd()), common_dir_name)
if common_dir_path not in sys.path:
    sys.path.append(common_dir_path)

def save_checkpoint(model, optimizer, epochs, epoch, loss, prefix, postfix, 
                    path, name):
    try:
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        file_name = f"{prefix}{name}_ep_{epoch+1}_of_{epochs}{postfix}.pth"
        file_path = os.path.join(path, file_name)

        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }
        torch.save(checkpoint, file_path)
        print(f"✅ Checkpoint saved: {file_path}")
        
    except RuntimeError as e:
        print(f"⚠️ [Disk Error] Fail to save! : {e}")
    except Exception as e:
        print(f"⚠️ Unknown error occurred: {e}")

def load_checkpoint(model, optimizer, file_path):
    if os.path.exists(file_path):
        checkpoint = torch.load(file_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f"🚀 Loaded checkpoint from '{file_path}' (Epoch {epoch})")
        return epoch, loss
    else:
        print("❌ No checkpoint found.")
        return 0, None