# Parallel-Computing-Application

## Project: Parallel image segmentation using k-means algorithm

Final project of Subject Parallel Computing Application.

Ho Chi Minh University Of Science.

Faculty of Information Technology.
# 

**Đồ Án Môn Học**: Lập trình song song ứng dụng

**Đề tài**: Song song và tối ưu hoá thuật toán k-means để phân vùng đối tượng trên video.

#

### Danh sách thành viên

| STT | MSSV    | Họ và Tên              | Tài khoản github                         |
| --- | ------- | ---------------------- | ---------------------------------------- |
| 1   | 1712117 | Nguyễn Huỳnh Thảo Nhi  | <https://github.com/NHTN>                |
| 2   | 1712713 | Lê Bá Quyền            | <https://github.com/LBQuyen>             |
| 3   | 1712775 | Nguyễn Lê Trường Thành | <https://github.com/noname-icecream0066> |

#
### Bảng kế hoạch

<table>
<thead>
  <tr>
   <th>Tuần</th>
   <th>Thời gian Bắt Đầu</th>
   <th>Thời Gian Kết Thúc</th>
   <th>Công Việc</th>
   <th>Thành Viên Thực Hiện</th>
   <th>Ghi Chú</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>1</td>
    <td>19/04/2021</td>
    <td>25/04/2021</td>
    <td>
      - Tìm hiểu, lựa chọn đề tài.
      <br>
      - Thống nhất đề tài đồ án.
     <br>
     - Tìm hiểu thuật toán K-means.
   </td>
   <td>Cả nhóm</td>
   <td></td>
  </tr>
  <tr>
    <td rowspan="2">2</td>
    <td rowspan="2">26/04/2021</td>
    <td rowspan="2">28/04/2021</td>
    <td> 
    - Mỗi thành viên cài đặt thuật toán tuần tự
    </td>
    <td>Cả nhóm</td>
   <td></td>
  </tr>

  <tr>
    <td> 
    - Merge code, soạn báo cáo, viết notebook
    </td>
    <td>Nguyễn Huỳnh Thảo Nhi</td>
   <td></td>
  </tr>

   <tr>
    <td>3</td>
    <td>29/04/2021</td>
    <td>29/04/2021</td>
    <td>Báo cáo tiến độ lần 01</td>
    <td>Lê Bá Quyền</td>
    <td></td>
   </tr>
   <tr>
   <td rowspan="4">4</td>
    <td >30/04/2021</td>
    <td>06/05/2021</td>
    <td>
    - Thay đổi đề tài cài đặt áp dụng trên video
   <br>
    - Tìm hiểu phương pháp áp dụng K-means trên video
   <br>
    - Tìm hiểu các phương pháp tối ưu K-means đối với bài toán phân đoạn
   </td>
    <td>Cả nhóm</td>
  </tr>

  <tr>
    <td>06/05/2021</td>
    <td>06/05/2021</td>
    <td>
    - Bổ sung, chỉnh sửa và hoàn thiện file README, Notebook
   </td>
    <td>Cả nhóm</td>
  </tr>

  <tr>
    <td>06/05/2021</td>
    <td>10/05/2021</td>
    <td>
    - Cài đặt thuật toán tuần tự đối với ảnh
    </br>
    - Cài đặt thuật toán tuần tự đối với video
   </td>
    <td>Cả nhóm</td>
  </tr>

  <tr>
    <td>11/05/2021</td>
    <td>12/05/2021</td>
    <td>
    - Thảo luận, tổng hợp tìm hiểu và cách cài đặt code
   </td>
    <td>Cả nhóm</td>
  </tr>


   <tr>
   <td>5</td>
    <td>13/05/2021</td>
    <td>13/05/2021</td>
    <td>Báo cáo tiến độ lần 02</td>
    <td>Nguyễn Lê Trường Thành</td>
  </tr>

  <tr>
   <td rowspan="4">6</td>
    <td >14/06/2021</td>
    <td>20/05/2021</td>
    <td>
    - Tìm hiểu các phương pháp tối ưu K-means tuần tự
   <br>
    - Tìm hiểu phương pháp áp dụng K-means trên video
   <br>
    - Tìm kiếm các ví dụ minh hoạ trực quan cho ứng dụng
   </td>
    <td>Cả nhóm</td>
  </tr>

  <tr>
    <td>21/05/2021</td>
    <td>23/05/2021</td>
    <td>
    - Bổ sung, chỉnh sửa và hoàn thiện file README, Notebook
   </td>
    <td>Nguyễn Huỳnh Thảo Nhi</td>
  </tr>

  <tr>
    <td>23/05/2021</td>
    <td>24/05/2021</td>
    <td>
    - Cải thiện thuật toán tuần tự
    <br>
    - Tối ưu code
   </td>
    <td>Cả nhóm</td>
  </tr>

  <tr>
    <td>25/05/2021</td>
    <td>26/05/2021</td>
    <td>
    - Thảo luận, tổng hợp code và merge code
   </td>
    <td>Cả nhóm</td>
  </tr>


  <tr>
   <td>7</td>
    <td>27/05/2021</td>
    <td>27/05/2021</td>
    <td>Báo cáo tiến độ lần 03</td>
    <td>Lê Bá Quyền</td>
  </tr>

  <tr>
   <td rowspan="5">8</td>
    <td>28/05/2021</td>
    <td>31/05/2021</td>
    <td>
    - Cập nhật numba trên thuật toán tuần tự
   </td>
    <td>Nguyễn Lê Trường Thành</td>
  </tr>

 <tr>
    <td>01/06/2021</td>
    <td>01/06/2021</td>
    <td>
    - Tìm hiểu phương pháp áp dụng K-means trên video
   <br>
    - Cập nhật hàm đánh giá thuật toán bằng thư viện OpenCV
   </td>
    <td>Nguyễn Huỳnh Thảo Nhi</td>
  </tr>

  <tr>
    <td>01/06/2021</td>
    <td>01/06/2021</td>
    <td>
    - Bổ sung, chỉnh sửa và hoàn thiện file README, Notebook
   </td>
    <td>Nguyễn Huỳnh Thảo Nhi</td>
  </tr>

  <tr>
    <td>01/06/2021</td>
    <td>08/06/2021</td>
    <td>
    - Cải thiện thuật toán tuần tự
    <br>
    - Cài đặt song song cho hàm chọn tâm của k-means
   </td>
    <td>Cả nhóm</td>
  </tr>

  <tr>
    <td>09/06/2021</td>
    <td>09/06/2021</td>
    <td>
    - Thảo luận, tổng hợp code và merge code
   </td>
    <td>Cả nhóm</td>
  </tr>

  <tr>
   <td>9</td>
    <td>10/06/2021</td>
    <td>10/06/2021</td>
    <td>Báo cáo tiến độ lần 04</td>
    <td>Nguyễn Huỳnh Thảo Nhi</td>
  </tr>

  <tr>
   <td rowspan="3">10</td>
    <td>11/06/2021</td>
    <td>23/06/2021</td>
    <td>
    - Tìm hiểu phương pháp tối ưu code song song trên video trên phiên bản hiện tại
    - Tìm hiểu và cài đặt phương pháp song song hoá các frames ảnh
    - Tìm hiểu phương pháp tối ưu elbow
   </td>
    <td>Cả nhóm</td>
  </tr>


  <tr>
    <td>23/06/2021</td>
    <td>23/06/2021</td>
    <td>
    - Thảo luận, tổng hợp code và merge code
   </td>
    <td>Cả nhóm</td>
  </tr>

  <tr>
    <td>24/06/2021</td>
    <td>24/06/2021</td>
    <td>Báo cáo tiến độ lần 05</td>
    <td>Nguyễn Lê Trường Thành</td>
  </tr>
</tbody>
</table>
