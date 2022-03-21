CREATE TABLE users (
   id SERIAL PRIMARY KEY,
   email TEXT,
   hashed_password TEXT,
   is_active BOOL DEFAULT true
);

CREATE INDEX idx_users_email ON users(email);

CREATE TABLE items (
   id SERIAL PRIMARY KEY,
   title TEXT,
   description TEXT,
   owner_id  INTEGER,

    CONSTRAINT fk_owner_id
        FOREIGN KEY(owner_id) 
	    REFERENCES users(id)
);

CREATE INDEX idx_items_title ON items(title);

CREATE INDEX idx_items_description ON items(description);
