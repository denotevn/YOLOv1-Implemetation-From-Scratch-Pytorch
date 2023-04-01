# YOLOv1-Implemetation-From-Scratch-Pytorch

### Architecture YOLOV1:
![Architecture](https://github.com/denotevn/YOLOv1-Implemetation-From-Scratch-Pytorch/blob/main/images/architectures.png)

### Result train with 20 epochs:
![result](https://github.com/denotevn/YOLOv1-Implemetation-From-Scratch-Pytorch/blob/main/images/Result.png)

### **Intersection Over Union (IoU)**
  + IOU (Intersection over Union) is a metric commonly used to evaluate the performance of object detection algorithms. It measures the overlap between the predicted bounding box and the ground truth bounding box of an object in an image.
  + IOU = (Area of Intersection) / (Area of Union)
  + The IOU value ranges from 0 to 1, where 0 indicates no overlap between the predicted and ground truth bounding boxes, and 1 indicates a perfect match.
### **Thuật toán Non-Maximum Suppression (NMS**
  + Thuật toán Non-Maximum Suppression (NMS) là một kỹ thuật được sử dụng trong bài toán object detection nhằm loại bỏ các bounding box trùng lặp và giữ lại các bounding box có độ tin cậy cao nhất.
  + Thuật toán NMS thường được áp dụng sau khi mô hình đã dự đoán được các bounding box chứa đối tượng trên ảnh đầu vào. Quá trình này tạo ra nhiều bounding box gần giống nhau tại một vị trí và kích thước khác nhau.
  + Trong quá trình loại bỏ, thuật toán sẽ giữ lại bounding box có độ tin cậy cao nhất và loại bỏ tất cả các bounding box khác có IOU với bounding box đó vượt qua ngưỡng xác định.
  + Kết quả của thuật toán NMS là tập hợp các bounding box không trùng lặp với độ tin cậy cao nhất, tạo ra một đầu ra chính xác hơn và dễ đọc hơn.
> Datasets **[Link](https://www.kaggle.com/code/kerneler/starter-pascalvoc-yolo-157cb3d6-0/data?select=train.csv)**