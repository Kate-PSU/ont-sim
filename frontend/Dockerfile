# frontend/Dockerfile
# Фронтенд на Vite + TypeScript
#
# Версия: 1.0
# Обновлено: 2026-04-06

# ========================================
# Stage 1: Builder
# ========================================
FROM node:22-alpine AS builder

# Рабочая директория
WORKDIR /app

# Копирование package.json
COPY package.json package-lock.json ./

# Установка зависимостей через npm
RUN npm ci

# Копирование исходного кода
COPY . .

# Сборка проекта
RUN npm run build

# ========================================
# Stage 2: Production
# ========================================
FROM nginx:alpine

# Копирование собранных файлов
COPY --from=builder /app/dist /usr/share/nginx/html

# Копирование nginx config
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Порты
EXPOSE 80

# Запуск nginx
CMD ["nginx", "-g", "daemon off;"]
