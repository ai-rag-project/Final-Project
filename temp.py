def count_words_in_file(file_path: str) -> int:
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            words = content.split()
            return len(words)
    except FileNotFoundError:
        print(f"[-] Error: The file '{file_path}' was not found.")
        return 0

if __name__ == "__main__":
    file_name = "test_data.txt"
    total_words = count_words_in_file(file_name)
    
    if total_words > 0:
        print(f"[+] Total words in '{file_name}': {total_words} words")