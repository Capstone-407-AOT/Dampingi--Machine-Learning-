cur = mysql.connection.cursor()
cur.execute("INSERT INTO MyUsers(firstName, lastName) VALUES (%s, %s)",
            (firstName, lastName))
mysql.connection.commit()
cur.close()
