Tuyệt vời! Lựa chọn bài toán số 1: **"Quản lý tài nguyên vô tuyến động (Dynamic Radio Resource Management - RRM)"** là một lựa chọn kinh điển và rất vững chắc. Đây là một vấn đề nền tảng trong mọi hệ thống không dây, và trong NTN nó trở nên đặc biệt thách thức, tạo ra nhiều cơ hội để đóng góp khoa học.

Bây giờ, chúng ta sẽ chính thức bước vào **Giai đoạn 2 (Xác định bài toán) và Giai đoạn 3 (Thực nghiệm)**. Tôi sẽ trình bày một cách chi tiết để bạn có thể hình dung rõ ràng từ ý tưởng đến lúc chạy mô phỏng.

---

### **Dự án nghiên cứu: Tối ưu hóa phân bổ Công suất và Băng thông trong mạng LEO bằng Deep Reinforcement Learning**

#### **1. Định nghĩa bài toán (Problem Statement)**

**Tên chính thức:** "Joint Power and Bandwidth Allocation for Downlink Transmission in a LEO Satellite Network using Deep Reinforcement Learning" (Phân bổ đồng thời Công suất và Băng thông cho đường xuống trong Mạng vệ tinh LEO sử dụng Học tăng cường sâu).

**Kịch bản (Scenario):**
*   Ta xét một hệ thống gồm **một vệ tinh LEO** đang bay qua một khu vực địa lý.
*   Vệ tinh này phục vụ **N người dùng mặt đất (UEs)** nằm trong vùng phủ của nó.
*   Vệ tinh có một tổng công suất phát là `P_max` và một tổng băng thông là `B_total` để chia sẻ cho N người dùng này.
*   **Mục tiêu:** Tại mỗi khoảng thời gian (timeslot), vệ tinh phải quyết định phân bổ bao nhiêu công suất và băng thông cho mỗi người dùng, nhằm **tối đa hóa một mục tiêu dài hạn**, ví dụ như tổng thông lượng của hệ thống trong khi vẫn đảm bảo sự công bằng (fairness) cho người dùng.

**Tại sao bài toán này khó và phù hợp với RL?**
*   **Động (Dynamic):** Vị trí của vệ tinh thay đổi, dẫn đến khoảng cách và góc tới người dùng thay đổi. Điều này làm cho **suy hao đường truyền (path loss)** và **chất lượng kênh (channel gain)** thay đổi liên tục.
*   **Trade-off (Sự đánh đổi):** Có một sự đánh đổi phức tạp. Dồn tài nguyên cho người dùng có kênh tốt sẽ tăng tổng thông lượng, nhưng lại gây "đói" tài nguyên cho người dùng có kênh yếu (vấn đề công bằng). Tăng công suất sẽ tăng tốc độ nhưng lại nhanh hết năng lượng.
*   **Không gian quyết định lớn:** Với N người dùng, việc quyết định một vector phân bổ `[p1, b1, p2, b2, ..., pN, bN]` là một bài toán tối ưu với không gian lời giải rất lớn.

#### **2. Xây dựng mô hình hệ thống (System Model)**

Đây là phần bạn sẽ viết trong mục "System Model" của paper.

*   **Mô hình kênh truyền (Channel Model):**
    *   Thông lượng (throughput) của người dùng `n` (`R_n`) có thể được tính bằng công thức Shannon-Hartley:
        `R_n = b_n * log2(1 + (p_n * g_n) / (N0 * b_n))`
    *   Trong đó:
        *   `b_n`: Băng thông được phân bổ cho người dùng `n`.
        *   `p_n`: Công suất được phân bổ cho người dùng `n`.
        *   `g_n`: Độ lợi kênh (channel gain) của người dùng `n`. Đây là thành phần **thay đổi theo thời gian**. Nó phụ thuộc vào suy hao đường truyền, shadowing, và fading.
        *   `N0`: Mật độ phổ nhiễu (noise power spectral density).
*   **Bài toán tối ưu hóa (Optimization Problem Formulation):**
    *   **Mục tiêu (Objective):** Tối đa hóa tổng thông lượng có trọng số công bằng (ví dụ, Proportional Fair):
        `Maximize: Σ [log(R_n)]` (Tổng log của thông lượng)
    *   **Ràng buộc (Constraints):**
        1.  `Σ p_n <= P_max` (Tổng công suất không vượt quá giới hạn).
        2.  `Σ b_n <= B_total` (Tổng băng thông không vượt quá giới hạn).
        3.  `p_n >= 0`, `b_n >= 0` (Công suất và băng thông không âm).

#### **3. Thiết kế giải pháp DRL (DRL-based Solution)**

Đây là trái tim của công trình nghiên cứu.

*   **Agent:** Bộ điều khiển RRM trên vệ tinh.
*   **State (S):** Trạng thái tại một thời điểm `t` phải chứa đủ thông tin để ra quyết định. Một thiết kế tốt là:
    *   `S_t = [g_1, g_2, ..., g_N, R_avg_1, R_avg_2, ..., R_avg_N]`
    *   `g_n`: Độ lợi kênh **hiện tại** của người dùng `n`.
    *   `R_avg_n`: Thông lượng **trung bình** của người dùng `n` trong một cửa sổ thời gian gần đây. Việc đưa thông tin này vào State giúp Agent học được về sự công bằng (nếu một người dùng có `R_avg` thấp, Agent có thể sẽ ưu tiên họ).
*   **Action (A):** Hành động tại thời điểm `t`. Đây là phần khó và có nhiều cách thiết kế:
    *   **Cách 1 (Không gian rời rạc - Discrete):** Chia các mức công suất và băng thông thành các mức rời rạc. Ví dụ, công suất có các mức {0.1W, 0.2W, ...} và băng thông có các mức {1MHz, 2MHz, ...}. Hành động là chọn một cặp (công suất, băng thông) cho mỗi người dùng. -> Không gian hành động bùng nổ rất nhanh với số lượng người dùng lớn.
    *   **Cách 2 (Không gian liên tục - Continuous):** Hành động là một vector `A_t` có `2N` chiều: `[p_1, ..., p_N, b_1, ..., b_N]`. Sau khi mạng nơ-ron xuất ra vector này, ta cần một bước "chuẩn hóa" (normalization) để đảm bảo nó thỏa mãn các ràng buộc (ví dụ, chia tỷ lệ để tổng công suất bằng `P_max`). -> Phù hợp với các thuật toán DRL cho không gian liên tục như **DDPG, PPO, SAC**.
*   **Reward (R):** Phần thưởng tại thời điểm `t` phải phản ánh mục tiêu của bài toán.
    *   `R_t = Σ [log(R_n)]` (Chính là hàm mục tiêu).
    *   Có thể thêm một thành phần phạt nếu vi phạm ràng buộc (mặc dù tốt hơn là xử lý trong bước Action).
*   **Lựa chọn thuật toán DRL:**
    *   **PPO (Proximal Policy Optimization)** là một lựa chọn rất mạnh mẽ và ổn định, hoạt động tốt cho cả không gian hành động rời rạc và liên tục. **Đây là gợi ý hàng đầu của tôi để bạn bắt đầu.**

#### **4. Cách triển khai thực nghiệm (Implementation Plan)**

*   **Ngôn ngữ và Thư viện:**
    *   **Ngôn ngữ:** **Python**.
    *   **Mô phỏng môi trường:** `NumPy` để tính toán các phương trình kênh, quản lý trạng thái.
    *   **Triển khai DRL:** `PyTorch` hoặc `TensorFlow` để xây dựng mạng nơ-ron. Bạn có thể sử dụng các thư viện cấp cao như `stable-baselines3` (dựa trên PyTorch) hoặc `tf-agents` (dựa trên TensorFlow) để không phải code lại PPO từ đầu. **Khuyến khích dùng `stable-baselines3` vì nó dễ sử dụng và có cộng đồng lớn.**
*   **Các bước thực hiện:**
    1.  **Code Môi trường (Environment):**
        *   Tạo một class `LeoEnv` tuân thủ giao diện của `gymnasium`.
        *   Class này phải có các hàm chính:
            *   `__init__()`: Khởi tạo các tham số (số người dùng N, P_max, B_total, quỹ đạo vệ tinh...).
            *   `reset()`: Đưa môi trường về trạng thái ban đầu (bắt đầu một episode mới). Trả về State đầu tiên.
            *   `step(action)`: Nhận một `action` từ Agent, tính toán trạng thái mới, tính toán `reward`, và trả về `(next_state, reward, done, truncated, info)`.
    2.  **Code Agent (Agent/Model):**
        *   Sử dụng `stable-baselines3`, bạn chỉ cần định nghĩa mô hình PPO.
        *   `model = PPO("MlpPolicy", env, verbose=1)`
        *   "MlpPolicy" nghĩa là mạng nơ-ron của bạn là một Multi-layer Perceptron.
    3.  **Huấn luyện (Training):**
        *   `model.learn(total_timesteps=100000)`
        *   Agent sẽ tương tác với môi trường `LeoEnv` hàng trăm nghìn lần để học chính sách tối ưu.
        *   Lưu lại mô hình đã huấn luyện: `model.save("ppo_leo_rrm")`.
    4.  **Đánh giá và So sánh (Evaluation & Baselines):**
        *   Tải mô hình đã huấn luyện và cho nó chạy trên các kịch bản thử nghiệm khác nhau.
        *   **Quan trọng:** Bạn phải so sánh giải pháp của mình với các thuật toán **baseline** để chứng tỏ nó hiệu quả. Các baseline tốt là:
            *   **Random Allocation:** Phân bổ công suất và băng thông một cách ngẫu nhiên. Đây là baseline thấp nhất.
            *   **Greedy (Tham lam):** Tại mỗi thời điểm, dồn hết tài nguyên cho người dùng có kênh tốt nhất `g_n`.
            *   **Round Robin (Xoay vòng):** Lần lượt phục vụ từng người dùng một.
*   **Các đồ thị cần vẽ cho paper:**
    *   Đồ thị **Hội tụ của phần thưởng (Reward Convergence):** Cho thấy phần thưởng trung bình tăng dần qua các vòng lặp huấn luyện, chứng tỏ Agent đang học.
    *   Đồ thị **So sánh hiệu năng:** So sánh tổng thông lượng (Total Throughput) và chỉ số công bằng (Jain's Fairness Index) của giải pháp DRL với các baseline.
    *   Đồ thị **Phân bổ tài nguyên động:** Vẽ biểu đồ thể hiện cách giải pháp DRL phân bổ công suất/băng thông cho các người dùng khác nhau theo thời gian.

---

### **Bắt đầu hành động**

Bây giờ bạn đã có một bản kế hoạch rất chi tiết. Bước tiếp theo là bắt tay vào code.

1.  **Cài đặt:** `pip install numpy gymnasium torch stable-baselines3`
2.  **Bắt đầu code:** Hãy bắt đầu với việc xây dựng class môi trường `LeoEnv`. Đây là phần quan trọng nhất. Đừng lo lắng về quỹ đạo vệ tinh phức tạp ngay, bạn có thể giả định `g_n` thay đổi ngẫu nhiên theo một phân phối nào đó ở mỗi `step` để đơn giản hóa lúc đầu.

Bạn thấy kế hoạch này thế nào? Có phần nào bạn cảm thấy chưa rõ hoặc cần làm sáng tỏ hơn không?
