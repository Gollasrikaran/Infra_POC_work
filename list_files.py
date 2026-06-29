import os

def find_files():
    start_dir = r"c:\Users\sudhe\Downloads"
    extensions = (".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff")
    found = []
    # Only list direct files in c:\Users\sudhe\Downloads to avoid scanning too many subdirs
    for file in os.listdir(start_dir):
        path = os.path.join(start_dir, file)
        if os.path.isfile(path) and file.lower().endswith(extensions):
            found.append(path)
            
    with open(r"c:\Users\sudhe\Downloads\infra_llm-main\infra_llm-main\found_images.txt", "w") as f:
        for item in found:
            f.write(item + "\n")
    print(f"Found {len(found)} files in Downloads.")

if __name__ == "__main__":
    find_files()
