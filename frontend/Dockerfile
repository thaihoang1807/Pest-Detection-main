# Sử dụng image Node.js siêu nhẹ
FROM node:18-alpine

# Cài đặt thư mục làm việc
WORKDIR /app

# Copy các file cấu hình và cài đặt thư viện
COPY package*.json ./
RUN npm install

# Copy toàn bộ source code frontend vào
COPY . .

# Mở port mặc định của Vite (thường là 5173)
EXPOSE 5173

# Lệnh chạy giao diện web
CMD ["npm", "run", "dev", "--", "--host"]