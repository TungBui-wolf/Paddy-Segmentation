# Paddy-Segmentation
## 1. Đặt vấn đề 
#### Giả sử khách hàng có 1 ảnh vệ tinh và 1 shape file chứa thông tin đo đạc ở 1 số vùng nhất định trên ảnh vệ tinh đó . 
#### Khách hàng muốn dựa vào thông tin đo đạc từ 1 phần nhỏ các khu vực đó, tiến hành dự đoán thông tin của toàn bộ ảnh vệ tinh (ở đây là dự đoán khu vực nào là đồng ruộng)
####
![Alt Text](http://https://drive.google.com/file/d/172RrSi4SbG3JJ-2mZkHiocajCi7n5Lnq/view?usp=sharing)

## 2. Hướng xử lí dữ liệu
Có 2 hướng xử lí : 
* Sử dụng U-net 
* Theo hướng pixel-based
#### Nhóm tập trung theo hướng pixel-based

## 3. Tiền xử lí dữ liệu
Từ shapefile và hình ảnh vệ tinh, xác định điểm dữ liệu 
#### => Tạo thành 1 file csv chứa 4787 điểm dữ liệu (pixel) đã được gán nhãn (paddy hoặc background)

## 4. Xây dựng model
Tách 4787 điểm dữ liệu thành 
* 3000 điểm dữ liệu train
* 1000 điểm dữ liệu validation
* 787 điểm dữ liệu test
#### Loss function : Binary Cross-Entropy
#### Optimal : Adam

## 5. Hậu xử lí
Xử lí các điểm dữ liệu có thể là lỗi, hoặc dự đoán sai, lệch khỏi cụm
