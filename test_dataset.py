from transformers import AutoTokenizer
from src.core.dtos import HateSpeechSample
from src.core.dataset import HateSpeechDataset


def test_dataset():
    print("--> Đang tải PhoBERT Tokenizer (lần đầu sẽ hơi lâu)...")
    # Tải bộ từ điển chuẩn của VinAI
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")

    # Giả lập dữ liệu đã clean
    fake_data = [
        HateSpeechSample(text="mày ngu quá", label="1"),
        HateSpeechSample(text="hôm nay trời đẹp", label="0")
    ]

    # Khởi tạo Dataset
    dataset = HateSpeechDataset(fake_data, tokenizer, max_len=10)

    # Lấy thử phần tử đầu tiên
    print("\n--> Lấy mẫu tại index 0:")
    sample = dataset[0]

    print("Text gốc:", fake_data[0].text)
    print("Input IDs (Số hóa):", sample['input_ids'])
    print("Attention Mask:", sample['attention_mask'])
    print("Label:", sample['labels'])

    print("\n--> KẾT LUẬN: Nếu thấy Input IDs là một danh sách các con số thì dataset ngon lành!")


if __name__ == "__main__":
    test_dataset()