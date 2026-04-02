```
[ BẮT ĐẦU THUẬT TOÁN SFS ]
      │
      ▼
┌─────────────────────────────────────────────────────┐
│ VÒNG LẶP: Đánh giá từng Feature ứng viên            │
│ (Ví dụ có 100 features cần test)                    │
└─┬───────────────────────────────────────────────────┘
  │
  ├─► Đánh giá Feature A:
  │     ├─►  Gọi hàm: Trộn & Cắt Data thành 5 phần
  │     ├─►  Train & Test Model trên 5 phần đó
  │     └─►  Ra điểm số (Điểm ảo do hên/xui bốc trúng data dễ/khó)
  │
  ├─► Đánh giá Feature B:
  │     ├─►  Gọi hàm: LẠI Trộn & Cắt Data mới
  │     ├─►  Train & Test Model trên 5 phần mới
  │     └─►  Ra điểm số
  │
  ... (Lặp lại việc trộn & cắt data 100 lần) ...

```

```
[ BẮT ĐẦU THUẬT TOÁN SFS ]
      │
      ▼
[  BƯỚC CHUẨN BỊ: CHIA DATA ĐÚNG 1 LẦN DUY NHẤT ]
(Tạo 5 Folds, lưu toàn bộ vị trí Train/Test vào RAM )
      │
      ▼
┌─────────────────────────────────────────────────────┐
│    VÒNG LẶP: Đánh giá từng Feature ứng viên         │
│ (Ví dụ có 100 features cần test)                    │
└─┬───────────────────────────────────────────────────┘
  │
  ├─► Đánh giá Feature A:
  │     ├─►  Móc bộ chia có sẵn trong RAM ra xài
  │     ├─►  Train & Test Model
  │     └─►  Ra điểm số (Chuẩn xác, công bằng)
  │
  ├─► Đánh giá Feature B:
  │     ├─►  Vẫn lấy bộ chia cũ trong RAM ra xài
  │     ├─►  Train & Test Model (Thi chung 1 đề)
  │     └─►  Ra điểm số (So sánh công bằng tuyệt đối)
  │
  ... (100 feature đều dùng chung 1 bộ chia cố định) ...
```
