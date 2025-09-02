# db_manager.py
import psycopg2
import pandas as pd
import json
import os
from logger_config import logger
from psycopg2 import sql

class DBManager:
    def __init__(self, db_params=None):
        if db_params is None:
            db_params = {
                'dbname': 'spectrum_data',
                'user': 'postgres',
                'password': '12345',
                'host': 'localhost',
                'port': '5432'
            }
        self.conn_params = db_params
        self.conn = None
        self.connect()
        self.create_table()

    def connect(self):
        try:
            # Преобразуем все строковые параметры в UTF-8
            clean_params = {}
            for key, value in self.conn_params.items():
                if isinstance(value, str):
                    clean_params[key] = value.encode('utf-8').decode('utf-8')
                else:
                    clean_params[key] = value
            
            self.conn = psycopg2.connect(**clean_params)
            logger.info("Успешное подключение к PostgreSQL")
        except Exception as e:
            logger.error(f"Ошибка подключения к PostgreSQL: {str(e)}")
            raise

    def create_table(self):
        """Создание таблицы graphs если она не существует"""
        try:
            with self.conn.cursor() as cursor:
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS graphs (
                        id SERIAL PRIMARY KEY,
                        image_name TEXT,
                        wavelength TEXT,
                        intensity TEXT,
                        min_peak TEXT,
                        max_peak TEXT
                    )
                """)
                self.conn.commit()
                logger.debug("Таблица graphs создана или уже существует")
        except Exception as e:
            logger.error(f"Ошибка при создании таблицы: {e}")
            self.conn.rollback()

    def save_sample(self, image_name, wavelength, intensity, min_peak, max_peak):
        """Сохранение образца в базу данных"""
        try:
            with self.conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO graphs (image_name, wavelength, intensity, min_peak, max_peak)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING id
                """, (image_name, json.dumps(wavelength), json.dumps(intensity), 
                     json.dumps(min_peak), json.dumps(max_peak)))
                record_id = cursor.fetchone()[0]
                self.conn.commit()
                logger.debug(f"Сохранена запись с id={record_id}, image_name={image_name}")
                return record_id
        except Exception as e:
            logger.error(f"Ошибка при сохранении записи: {e}")
            self.conn.rollback()
            raise

    def get_all_data(self):
        """Получение всех данных из таблицы graphs"""
        try:
            with self.conn.cursor() as cursor:
                cursor.execute("SELECT * FROM graphs")
                rows = cursor.fetchall()
                logger.debug(f"Получено {len(rows)} записей из базы данных")
                
                if rows:
                    columns = [desc[0] for desc in cursor.description]
                    df = pd.DataFrame(rows, columns=columns)
                    # Преобразование JSON строк обратно в списки
                    for col in ['wavelength', 'intensity', 'min_peak', 'max_peak']:
                        df[col] = df[col].apply(json.loads)
                    logger.debug(f"Обработанные данные: {df.to_dict(orient='records')}")
                    return df
                return pd.DataFrame(columns=['id', 'image_name', 'wavelength', 'intensity', 'min_peak', 'max_peak'])
        except Exception as e:
            logger.error(f"Ошибка при получении данных: {e}")
            return pd.DataFrame()

    def get_page_data(self, page, rows_per_page):
        """Получение данных для конкретной страницы"""
        try:
            offset = (page - 1) * rows_per_page
            with self.conn.cursor() as cursor:
                cursor.execute("""
                    SELECT * FROM graphs 
                    ORDER BY id 
                    LIMIT %s OFFSET %s
                """, (rows_per_page, offset))
                rows = cursor.fetchall()
                logger.debug(f"Получено {len(rows)} записей для страницы {page}")
                
                if rows:
                    columns = [desc[0] for desc in cursor.description]
                    # Создаем список словарей с данными
                    data = []
                    for row in rows:
                        record = dict(zip(columns, row))
                        # Преобразование JSON строк обратно в списки
                        for col in ['wavelength', 'intensity', 'min_peak', 'max_peak']:
                            record[col] = json.loads(record[col])
                        data.append(record)
                    
                    # Создаем DataFrame из списка словарей
                    df = pd.DataFrame(data)
                    return df
                return pd.DataFrame(columns=['id', 'image_name', 'wavelength', 'intensity', 'min_peak', 'max_peak'])
        except Exception as e:
            logger.error(f"Ошибка при получении данных страницы: {e}")
            return pd.DataFrame()

    def count_total_records(self):
        """Подсчёт общего числа записей в таблице graphs"""
        try:
            with self.conn.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) FROM graphs")
                total = cursor.fetchone()[0]
                logger.debug(f"Общее количество записей в базе: {total}")
                return total
        except Exception as e:
            logger.error(f"Ошибка при подсчёте записей: {e}")
            return 0

    def update_record(self, record_id, image_name=None, wavelength=None, 
                     intensity=None, min_peak=None, max_peak=None):
        """Обновление записи в базе данных"""
        updates = []
        params = []
        
        if image_name is not None:
            updates.append("image_name = %s")
            params.append(image_name)
        if wavelength is not None:
            updates.append("wavelength = %s")
            params.append(json.dumps(wavelength))
        if intensity is not None:
            updates.append("intensity = %s")
            params.append(json.dumps(intensity))
        if min_peak is not None:
            updates.append("min_peak = %s")
            params.append(json.dumps(min_peak))
        if max_peak is not None:
            updates.append("max_peak = %s")
            params.append(json.dumps(max_peak))
            
        if updates:
            params.append(record_id)
            query = f"UPDATE graphs SET {', '.join(updates)} WHERE id = %s"
            
            try:
                with self.conn.cursor() as cursor:
                    cursor.execute(query, params)
                    self.conn.commit()
                    logger.debug(f"Запись с id={record_id} обновлена")
            except Exception as e:
                logger.error(f"Ошибка при обновлении записи: {e}")
                self.conn.rollback()
                raise


    # db_manager.py (добавление метода get_record_by_id)
    def get_record_by_id(self, record_id):
        """Получение записи по ID из базы данных."""
        try:
            with self.conn.cursor() as cursor:
                cursor.execute("SELECT * FROM graphs WHERE id = %s", (record_id,))
                row = cursor.fetchone()
                if row:
                    columns = [desc[0] for desc in cursor.description]
                    data = dict(zip(columns, row))
                    # Преобразование JSON строк обратно в списки
                    for col in ['wavelength', 'intensity', 'min_peak', 'max_peak']:
                        data[col] = json.loads(data[col])
                    return data
                return None
        except Exception as e:
            logger.error(f"Ошибка при получении записи: {e}")
            return None

    def delete_by_id(self, record_id):
        """Удаление записи по ID"""
        logger.debug(f"Попытка удаления для id={record_id}")
        try:
            with self.conn.cursor() as cursor:
                # Получаем данные перед удалением для логов
                cursor.execute("SELECT * FROM graphs WHERE id = %s", (record_id,))
                rows_to_delete = cursor.fetchall()
                logger.debug(f"Найдено строк для удаления: {len(rows_to_delete)}")
                if rows_to_delete:
                    logger.debug(f"Данные перед удалением: {rows_to_delete}")
                
                # Выполняем удаление
                cursor.execute("DELETE FROM graphs WHERE id = %s", (record_id,))
                rows_deleted = cursor.rowcount
                self.conn.commit()
                logger.debug(f"Удалено строк: {rows_deleted}")
                
                # В PostgreSQL нет необходимости сбрасывать sequence, как в SQLite
                return rows_deleted
        except Exception as e:
            logger.error(f"Ошибка при удалении записи: {e}")
            self.conn.rollback()
            raise

    def close(self):
        """Закрытие соединения с базой данных"""
        if self.conn and not self.conn.closed:
            self.conn.close()
            logger.info("Соединение с PostgreSQL закрыто")

