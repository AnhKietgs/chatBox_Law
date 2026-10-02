FROM node:20.20-alpine AS build
WORKDIR /app
COPY package.json package-lock.json ./
# Use the lockfile so Docker builds the exact Vite/TypeScript versions that
# were tested locally instead of resolving the moving "latest" tags.
RUN npm ci
COPY . .
RUN npm run build
FROM nginx:1.27-alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
