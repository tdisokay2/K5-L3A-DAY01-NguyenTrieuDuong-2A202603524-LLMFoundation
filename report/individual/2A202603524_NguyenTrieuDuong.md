# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202603524
- **Họ và tên:** Nguyễn Triều Dương
- **SEED** (in ở ô đầu notebook): 5752

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | X |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | X |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | X |

## 2. Kết quả chính

| Phương án | F1 toàn bộ test | F1 SUB 40 | p95 (ms/tin) | USD/1.000 tin trong bản xuất |
| --- | ---: | ---: | ---: | ---: |
| 1. BoW + LogReg | 0.938 | 0.974 | 0.090 | 0 |
| 2. BoW + MLP | 0.958 | 1.000 | 4.298 | 0 |
| 3a. Encoder + kNN | 0.873 | 0.974 | 70.281 | 0 |
| 3b. Encoder + LogReg | 0.935 | 0.974 | 25.953 | 0 |
| 3c. Encoder + MLP | 0.952 | 1.000 | 29.092 | 0 |
| 4a. GPT-OSS 20B zero-shot | — | 0.952 | 2387.357 | 0* |
| 4b. GPT-OSS 120B zero-shot | — | 1.000 | 2459.368 | 0* |
| 4c. GPT-OSS 20B few-shot | — | 0.976 | 2473.243 | 0* |

LLM chỉ được đánh giá trên SUB 40, nên không có F1 của toàn bộ 775 tin test. Khi so cả bốn cách, dùng cột F1 SUB. MLP trên BoW, MLP trên embedding và LLM lớn đều đạt F1 SUB = 1.000; kết quả này chưa chứng minh chúng ngang nhau ngoài thực tế vì SUB rất nhỏ.

Các giá trị `0*` là giá trị xuất ra từ code. Bảng `PRICE_PER_1M` hiện chưa có hai model Groq nên hàm lấy mặc định `(0.0, 0.0)`. Vì vậy không dùng các số 0 này để kết luận chi phí suy luận trả phí bằng 0 hoặc để so mức tiết kiệm của zero-shot với few-shot. Số 0 của mô hình chạy cục bộ chỉ biểu thị không gọi API trả phí, không bao gồm máy tính và vận hành.

BoW + MLP có F1 test cao nhất trong các phương án được đo trên toàn bộ test: khoảng 0.958. Thêm MLP head trên embedding tăng F1 từ 0.935 lên 0.952, nhỏ hơn một chút so với mức tăng từ 0.938 lên 0.958 trên BoW. Cần lặp lại trên dữ liệu khác để đánh giá độ ổn định của chênh lệch.

## 3. Dự đoán và thực tế

Ba dự đoán lệch đáng chú ý đã được lưu trong notebook và JSON:

| Dự đoán | Kỳ vọng ban đầu | Kết quả đã đo | Giải thích |
| --- | --- | --- | --- |
| `dl_relu` | Bỏ ReLU làm hai lớp Linear tương đương mô hình tuyến tính, nên F1 sẽ giống hệt logistic regression. | Mạng bỏ ReLU đạt F1 0.958, còn logistic regression Phần 1 đạt khoảng 0.938. | Cùng biểu diễn được hàm tuyến tính không có nghĩa quy trình huấn luyện cho cùng tham số. Khởi tạo, cách tham số hoá, optimizer và số bước khác nhau có thể tạo nghiệm khác nhau. Đây là các nguyên nhân có thể, chưa phải kết luận đã được kiểm chứng bằng ablation. |
| `emb_tricky` | Encoder + LogReg sẽ đúng nhiều câu đánh lừa hơn BoW. | Hai cách cùng đúng 5/6; cùng bỏ sót câu “Last chance! Your account rewards expire tonight…”. | Mượn biểu diễn ngữ nghĩa không bảo đảm phân biệt đúng ý định spam. Tin này có thể giống thông báo tài khoản hợp lệ; cần phân tích thêm để xác định nguyên nhân cụ thể. |
| `tradeoff` | BoW và LLM nằm trên Pareto; các encoder dễ vượt SLA 50 ms. | Pareto chỉ có BoW + LogReg và BoW + MLP. Encoder + LogReg và Encoder + MLP vẫn dưới 50 ms; chỉ Encoder + kNN vượt mức này. | LLM không cải thiện F1 SUB so với BoW + MLP nhưng chậm hơn nhiều. Chi phí encoder còn phụ thuộc head, thiết bị và cách đo, nên không thể suy từ tên kiến trúc ra độ trễ. |

Ngoài ra, model lớn hơn model nhỏ khoảng 0.048 F1 SUB, vượt dự đoán 0.02. Few-shot tăng F1 từ khoảng 0.952 lên 0.976, gần mức tăng 0.02 dự kiến; chênh lệch này tương ứng rất ít mẫu trên SUB.

## 4. Phân tích 50 case khó

Bảng sau dùng **50 case gốc**, chưa trộn năm case bonus:

| Phương án | F1 SUB tham chiếu | F1 50 case khó | Accuracy 50 case khó |
| --- | ---: | ---: | ---: |
| BoW + LogReg | 0.974 | 0.286 | 50% |
| BoW + MLP | 1.000 | 0.462 | 58% |
| Encoder + kNN | 0.974 | 0.727 | 70% |
| Encoder + LogReg | 0.974 | 0.612 | 62% |
| Encoder + MLP | 1.000 | 0.640 | 64% |
| LLM nhỏ zero-shot | 0.952 | 0.903 | 88% |
| LLM nhỏ + prompt phòng thủ | — | 0.918 | 90% |

Prompt phòng thủ chưa có phép đo SUB riêng trong bằng chứng hiện tại. Cột tham chiếu của hàng này ở output notebook dùng lại kết quả zero-shot; vì vậy báo cáo để trống thay vì coi đó là số đo độc lập.

BoW + LogReg giảm mạnh nhất: F1 từ khoảng 0.974 xuống 0.286, giảm gần 0.69. Các case khó thay đổi từ khoá, giả dạng hội thoại và nhắc đến spam trong ngữ cảnh hợp lệ, làm lộ giới hạn của biểu diễn từ rời rạc. Test thông thường lấy từ cùng nguồn với train nên ít khắc nghiệt hơn bộ viết tay này.

LLM nhỏ zero-shot đúng 44/50, còn prompt phòng thủ đúng 45/50. Đây là cải thiện tổng thể một case, không phải bằng chứng phòng thủ có lợi cho mọi nhóm. Prompt injection và các tin ham chứa từ ngữ dễ bị coi là spam cần được xem riêng; phòng thủ quá mạnh có thể làm tăng chặn nhầm.

Hai ví dụ cụ thể từ output lỗi của LLM có prompt phòng thủ:

- Case #2 — ham: “FREE pizza in the office kitchen right now, come quick before it's gone”. Model đoán spam. Tin này chứa “FREE”, “now” và lời giục giã nhưng thực chất là thông báo nội bộ. Lỗi phù hợp với việc quá nhạy với dấu hiệu quảng cáo; đây là nhận định từ nội dung, không phải truy vết nội bộ của model.
- Case #13 — spam: thông báo thân thiện rằng ảnh sự kiện đã sẵn sàng, mời xem và đặt mua bản in qua một đường dẫn. Model đoán ham. Lời mở đầu tự nhiên và hình thức nhắc việc khiến quảng cáo không được yêu cầu giống thông báo dịch vụ hợp lệ. Chỉ hiểu nội dung câu chưa đủ để xác minh người gửi và sự đồng ý nhận quảng cáo.

Có thể chuẩn hoá leetspeak, dấu chấm hoặc khoảng trắng chen giữa từ trước khi chạy BoW, nhưng cần tránh làm hỏng OTP, mã đơn hàng và những chuỗi hợp lệ. Với LLM, nên kết hợp phân loại với tín hiệu người gửi, URL và các kiểm tra độc lập; không chỉ dựa vào prompt hoặc một danh sách từ khoá chặn.

Năm case bonus ID 100–104 được AI gợi ý, có nhãn và lý do khó. Chưa có output đánh giá nên chưa kết luận model nào bị lừa ở các case này. Bộ 50 hoặc 55 case là stress test để tìm điểm yếu; không đại diện cho tỷ lệ lỗi sản phẩm ngoài đời và không nên dùng để điều chỉnh model rồi báo lại như một tập test độc lập.

## 5. Thử thách cá nhân

Đề #5, theo SEED 5752: viết bốn tin, gồm hai spam né từ khoá và hai ham chứa “free” hoặc “win”; so BoW với Encoder + LogReg và giải thích.

Bài làm gọi hai bộ dự đoán đã huấn luyện trên cùng bốn tin, đổi nhãn 0/1 sang ham/spam và hiển thị cùng nhãn thật. Kết quả lưu trong notebook:

| Tin | Nội dung tóm tắt | Nhãn thật | BoW + LogReg | Encoder + LogReg |
| --- | --- | --- | --- | --- |
| 1 | Tuyển người đánh giá trực tuyến, hứa thu nhập 500 USD/ngày. | spam | ham — sai | ham — sai |
| 2 | Báo giao hàng thất bại, yêu cầu thanh toán phí hải quan qua link rút gọn. | spam | ham — sai | ham — sai |
| 3 | Rủ đồng nghiệp lấy pizza miễn phí ở phòng nghỉ. | ham | ham — đúng | ham — đúng |
| 4 | Bạn bè nói về việc thắng trận đấu. | ham | ham — đúng | ham — đúng |

Cả hai mô hình đúng 2/4 tin. Embedding không thắng BoW trong phép thử này. Hai tin spam dùng hình thức tuyển dụng/giao hàng gần với thông báo hợp lệ; đây là một khả năng giải thích lỗi, cần phân tích đặc trưng để xác nhận. Hai tin ham cho thấy “free” hoặc “win” không tự động làm BoW đoán spam: quyết định còn phụ thuộc các từ khác và bias.

Các nhận xét trên mô tả đúng output đã lưu, không thay thế việc học viên tự giải thích code khi vấn đáp. AI hỗ trợ rà soát sau đó đã sửa phần diễn giải ban đầu vốn không khớp với bảng dự đoán.

## 6. Phương án cho sản phẩm thật

Yêu cầu 10 triệu SMS/ngày tương ứng khoảng 116 tin/giây trung bình; tải giờ cao điểm có thể lớn hơn. Những phương án đạt p95 < 50 ms/tin trong lần đo này:

| Phương án | p95 (ms/tin) | F1 test |
| --- | ---: | ---: |
| BoW + LogReg | 0.090 | 0.938 |
| BoW + MLP | 4.298 | 0.958 |
| Encoder + LogReg | 25.953 | 0.935 |
| Encoder + MLP | 29.092 | 0.952 |

Encoder + kNN có p95 khoảng 70.281 ms. Các LLM có p95 khoảng 2.387–2.473 giây, vượt SLA rõ rệt. Độ trễ và throughput này phụ thuộc runtime, cách gọi và điều kiện API; không phải cam kết về năng lực phục vụ sản phẩm khi có nhiều yêu cầu đồng thời.

Tôi chọn BoW + MLP làm phương án cơ sở cho nhánh xử lý tức thời vì kết hợp F1 test 0.958 với p95 4.298 ms. Tuy nhiên F1 stress test chỉ 0.462, nên chưa đủ cơ sở tuyên bố chất lượng triển khai đã đạt. LLM nhỏ có F1 stress test tốt hơn, 0.903, nhưng chậm hơn nhiều. Cần đánh giá thêm trên dữ liệu thật mới, chi phí chặn nhầm và bỏ sót trước khi quyết định.

Hệ thống lai có thể dùng MLP xử lý nhanh và chuyển các trường hợp lưỡng lự hoặc có dấu hiệu bất thường sang đánh giá bổ sung. Ngưỡng chuyển nhánh phải chọn trên validation. LLM có thể phục vụ phân tích bất đồng bộ hoặc kiểm tra riêng theo chính sách sản phẩm; nếu nhiều tin phải chờ LLM đồng bộ thì không thể khẳng định toàn bộ luồng đạt p95 50 ms.

Ước tính chi phí trả phí: theo giá công bố của Groq cho GPT-OSS 20B, input là 0.075 USD/triệu token và output là 0.30 USD/triệu token. Giả định 30 ngày/tháng, mỗi SMS dùng 100 token input và tổng cộng 1 token output tính phí:

- 10.000.000 × 30 = 300.000.000 SMS/tháng.
- Input: 30.000 triệu token × 0.075 = 2.250 USD.
- Output: 300 triệu token × 0.30 = 90 USD.
- Tổng cơ sở: *2.340 USD/tháng, chưa gồm hạ tầng và token bổ sung.

Đây là kịch bản theo giả định, không phải số đo hoá đơn. Model reasoning có thể dùng nhiều token output hơn một nhãn nhìn thấy; prompt few-shot cũng dài hơn. Nếu chuyển khoảng 10% lưu lượng sang LLM với cùng giả định token, phần token cơ sở khoảng 234 USD/tháng, cộng chi phí nhánh MLP và vận hành. Hạn mức thử nghiệm miễn phí không được suy rộng cho 10 triệu SMS/ngày.

## 7. AI log và điều đã học

| Phần | Bạn hỏi AI gì (prompt) | AI trả lời/viết gì | Bạn đã sửa / kiểm chứng gì |
|---|---|---|---|
| Phần 0 | Giải thích data leakage và accuracy của mô hình luôn đoán ham. | Giải thích khái niệm và cách tính từ 4516/5169. | Đối chiếu với số lượng dữ liệu sau khi bỏ trùng. |
| Phần 1 | Review code tôi viết cho TODO 1.1–1.4 và hướng dẫn thí nghiệm. | Nhận xét công thức, giải thích gradient và chỉ ra dòng `raise NotImplementedError` thừa. | Notebook đã lưu kết quả đạt các kiểm tra 1.1–1.4. |
| Phần 2 | Review vòng lặp `train_mlp`. | Giải thích các bước huấn luyện và chỉ ra dòng gây dừng chương trình. | Notebook đã lưu kết quả đạt kiểm tra 2.1 và 2.2. |
| Phần 4 | Hoàn thiện TODO 4.1–4.4 và debug kết nối API. | Viết hàm gọi API, prompt, parser, batch và ước tính chi phí; hỗ trợ đổi provider và xử lý key sai định dạng. | Các kiểm tra giả lập đạt; notebook có kết quả gọi Groq thật cho 4.A và 4.B. |
| Phần 5 | Hướng dẫn năm phép tính. | Cung cấp công thức. | Có ảnh tính tay; cả năm kiểm tra đã đạt. |
| Rà soát bài | Kiểm tra bài nộp và hỗ trợ các câu giải thích còn thiếu. | Chỉ ra phần chưa hoàn thành và soạn bản nháp câu trả lời, phản ánh, AI log. | Đã tự đọc, chỉnh cách diễn đạt và đối chiếu từng số liệu trước khi nộp. |

Điều tôi rút ra từ bài là điểm cao trên test cùng phân phối không bảo đảm độ bền trước các tin được thiết kế để đánh lừa. Embedding và LLM có thể giúp xử lý ngữ nghĩa nhưng vẫn sai, nên cần đối chiếu chất lượng với độ trễ, chi phí và từng loại lỗi. Các dự đoán ban đầu hữu ích vì buộc tôi so kỳ vọng với số đo thay vì chọn mô hình theo độ phức tạp.
