CREATE DATABASE login_app;
GO

USE login_app;
GO

CREATE TABLE users (
    id INT IDENTITY(1,1) PRIMARY KEY,
    username NVARCHAR(50) UNIQUE NOT NULL,
    password NVARCHAR(255) NOT NULL
);


select * from users

ALTER TABLE users ADD role VARCHAR(20) DEFAULT 'user';

INSERT INTO users (username, password, role) VALUES ('admin1', '12345', 'admin');


ALTER TABLE users
ADD last_updated DATETIME NULL;

CREATE TABLE users (
    id INT IDENTITY(1,1) PRIMARY KEY,
    username NVARCHAR(100) NOT NULL UNIQUE,
    password NVARCHAR(255) NOT NULL,
    role NVARCHAR(50) NOT NULL DEFAULT 'user',
    bio NVARCHAR(MAX) NULL,
    avatar VARBINARY(MAX) NULL,          -- menyimpan file gambar sebagai bytes
    avatar_filename NVARCHAR(255) NULL,  -- menyimpan nama file dan extension
    created_at DATETIME NOT NULL DEFAULT GETDATE(),
    last_updated DATETIME NULL
);

INSERT INTO users (username, password, role)
VALUES ('admin', 'admin123', 'admin');

UPDATE users
SET role = 'admin'       -- role baru yang ingin kamu tetapkan
WHERE username = 'hansen';  -- username target yang ingin diubah


ALTER TABLE users
ADD is_active BIT NOT NULL DEFAULT 1;

CREATE TABLE datasets (
    id INT IDENTITY(1,1) PRIMARY KEY,
    dataset_name NVARCHAR(255) NOT NULL,
    description NVARCHAR(MAX) NULL,
    label NVARCHAR(50) NOT NULL, -- Keseluruhan / Kecamatan / Kabupaten / Provinsi
    file_content VARBINARY(MAX) NOT NULL,
    file_name NVARCHAR(255) NOT NULL,
    file_type NVARCHAR(50) NOT NULL,
    uploaded_by NVARCHAR(100) NOT NULL,
    uploaded_at DATETIME DEFAULT GETDATE()
);

select * from datasets

CREATE TABLE login_logs (
    id INT IDENTITY(1,1) PRIMARY KEY,
    username NVARCHAR(100) NOT NULL,
    role NVARCHAR(50) NOT NULL,
    login_time DATETIME DEFAULT GETDATE()
);

ALTER TABLE datasets ADD is_selected BIT DEFAULT 1;

