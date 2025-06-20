#----------------------------------载入包-----------------------------------------
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy import create_engine, Column, Integer, String, func, DateTime, Date, cast, desc
from sqlalchemy import Boolean, SmallInteger, TEXT, Float, ForeignKey, and_, or_, case
from sqlalchemy.orm import sessionmaker, relationship, declarative_base, Session
from sqlalchemy.sql import text
from datetime import datetime, date
from pydantic import BaseModel, EmailStr, constr, Field, validator, field_validator
from typing import Optional, List, Annotated
from enum import Enum
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from fastapi.responses import JSONResponse
from scipy.stats import pearsonr
from itertools import combinations
from collections import defaultdict
from fastapi import Query



#---------------------------------数据库配置----------------------------------------
DATABASE_URL = "postgresql://postgres:postgres@localhost/smart_home"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()



# ------------------------- SQLAlchemy模型（添加Model后缀） -------------------------
# 用户
class UserModel(Base):
    __tablename__ = 'users'
    user_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    username = Column(String(30), unique=True, nullable=False)
    password = Column(String(255), nullable=False)
    gender = Column(String(10))
    birth_date = Column(Date)
    phone_number = Column(String(20), unique=True, nullable=False)
    house_area = Column(Float)
    email = Column(String(100), unique=True, nullable=False)

# 设备信息
class DeviceModel(Base):
    __tablename__ = "devices"
    device_id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(30), nullable=False)
    type = Column(String(10), nullable=False)
    location = Column(String(50), nullable=False)
    installation_time = Column(DateTime, nullable=False, server_default=func.now())

# 设备使用记录
class DeviceUsageRecordModel(Base):
    __tablename__ = 'device_usage_records'
    record_id = Column(Integer, primary_key=True, index=True)
    device_id = Column(Integer, ForeignKey('devices.device_id', ondelete="CASCADE"), nullable=False)
    user_id = Column(Integer, ForeignKey('users.user_id', ondelete="CASCADE"), nullable=False)
    start_time = Column(DateTime, nullable=False, server_default=func.now())
    end_time = Column(DateTime)
    temperature = Column(Float)
    energy_consumption = Column(Float, nullable=False)

# 安防事件
class SecurityEventModel(Base):
    __tablename__ = "security_events"
    event_id = Column(Integer, primary_key=True, autoincrement=True)
    device_id = Column(Integer, ForeignKey('devices.device_id', ondelete="CASCADE"), nullable=False)
    event_type = Column(String(20), nullable=False)
    occurrence_date = Column(DateTime, nullable=False, server_default=func.current_timestamp())
    severity = Column(String(20), nullable=False)
    is_resolved = Column(Boolean, nullable=False, default=False)
    description = Column(TEXT)

# 用户反馈
class UserFeedbackModel(Base):
    __tablename__ = "user_feedbacks"
    feedback_id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('users.user_id', ondelete="CASCADE"), nullable=False)
    device_id = Column(Integer, ForeignKey('devices.device_id', ondelete="CASCADE"), nullable=False)
    satisfaction_rating = Column(Integer, nullable=False)
    feedback_text = Column(TEXT)
    feedback_date = Column(DateTime, nullable=False, server_default=func.current_timestamp())



# ------------------------- Pydantic模型（添加Response后缀） -------------------------
class UserBase(BaseModel):
    username: Annotated[str, Field(min_length=4, max_length=30)]
    email: EmailStr
    phone_number: str
    gender: Optional[str] = Field(None, max_length=10)
    birth_date: Optional[date] = None
    house_area: float = Field(..., gt=0, lt=1000)

    @field_validator('username')
    def validate_username(cls, v):
        if not re.match(r'^[a-zA-Z][a-zA-Z0-9_]*$', v):
            raise ValueError('用户名必须以字母开头，仅包含字母、数字和下划线')
        return v

    @field_validator('phone_number')
    def validate_phone_number(cls, v):
        if not re.match(r'^[0-9]{11}$', v):
            raise ValueError('手机号必须为11位数字')
        return v

class UserCreate(UserBase):
    password: str = Field(..., min_length=6, max_length=16)

    @field_validator('password')
    def validate_password(cls, v):
        if not re.match(r'^[a-zA-Z][a-zA-Z0-9_]*$', v):
            raise ValueError('密码必须以字母开头，仅包含字母、数字和下划线')
        return v

class UserResponse(UserBase):
    user_id: int
    class Config:
        from_attributes = True

class DeviceType(str, Enum):
    air_conditioner = "空调"
    lighting = "照明"
    security = "安防"
    socket = "插座"
    entertainment = "娱乐"
    water_heater = "热水器"
    sensor = "传感器"
    appliance = "家电"
    curtain = "窗帘"
    other = "其他"

class Location(str, Enum):
    living_room = "客厅"
    master_bedroom = "主卧"
    secondary_bedroom = "次卧"
    study = "书房"
    balcony = "阳台"
    kitchen = "厨房"
    bathroom = "浴室"
    main_door = "大门"
    other = "其他"

class DeviceBase(BaseModel):
    name: constr(min_length=2, max_length=30)
    type: DeviceType
    location: Location
    installation_time: datetime

class DeviceCreate(DeviceBase):
    pass

class DeviceResponse(DeviceBase):
    device_id: int
    class Config:
        from_attributes = True

class DeviceUsageRecordBase(BaseModel):
    device_id: int
    user_id: int
    start_time: datetime
    end_time: Optional[datetime] = None
    temperature: Optional[float] = None
    energy_consumption: float = Field(..., ge=0)

class DeviceUsageRecordCreate(DeviceUsageRecordBase):
    pass

class DeviceUsageRecordResponse(DeviceUsageRecordBase):
    record_id: int
    class Config:
        from_attributes = True

class EventType(str, Enum):
    lock_abnormal = "门锁异常"
    motion_detected = "移动检测"
    smoke_alarm = "烟雾警报"
    intrusion_alarm = "入侵警报"
    doorbell_pressed = "门铃按动"
    water_leak = "水浸检测"
    glass_break = "玻璃破碎"
    emergency_button = "紧急按钮"
    temp_abnormal = "温度异常"
    wrong_password = "密码错误"
    wrong_fingerprint = "指纹错误"
    co_alarm = "一氧化碳警报"
    device_offline = "设备离线"
    low_battery = "电池电量低"
    illegal_entry = "非法闯入"
    other = "其它"

class SeverityLevel(str, Enum):
    low = "Low"
    medium = "Medium"
    high = "High"

class SecurityEventBase(BaseModel):
    device_id: int
    event_type: EventType
    severity: SeverityLevel
    is_resolved: bool = False
    description: Optional[str] = None

class SecurityEventCreate(SecurityEventBase):
    pass

class SecurityEventResponse(SecurityEventBase):
    event_id: int
    occurrence_date: datetime
    class Config:
        from_attributes = True

class UserFeedbackBase(BaseModel):
    user_id: int
    device_id: int
    satisfaction_rating: int = Field(..., ge=1, le=10)
    feedback_text: Optional[str] = None

class UserFeedbackCreate(UserFeedbackBase):
    pass

class UserFeedbackResponse(UserFeedbackBase):
    feedback_id: int
    feedback_date: datetime
    class Config:
        from_attributes = True



# ------------------------- CRUD操作 -------------------------------------
# 用户
def get_user(db: Session, user_id: int):
    return db.query(UserModel).filter(UserModel.user_id == user_id).first()

def get_users(db: Session, skip: int = 0, limit: int = 100):
    return db.query(UserModel).offset(skip).limit(limit).all()

def create_user(db: Session, user: UserCreate):
    db_user = UserModel(**user.dict())
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def update_user(db: Session, user_id: int, user: UserBase):
    db_user = db.query(UserModel).filter(UserModel.user_id == user_id).first()
    if db_user:
        for key, value in user.dict(exclude_unset=True).items():
            setattr(db_user, key, value)
        db.commit()
        db.refresh(db_user)
    return db_user

def delete_user(db: Session, user_id: int):
    db_user = db.query(UserModel).filter(UserModel.user_id == user_id).first()
    if db_user:
        db.delete(db_user)
        db.commit()
    return db_user

# 设备
def get_device(db: Session, device_id: int):
    return db.query(DeviceModel).filter(DeviceModel.device_id == device_id).first()

def get_devices(db: Session, skip: int = 0, limit: int = 100):
    return db.query(DeviceModel).offset(skip).limit(limit).all()

def create_device(db: Session, device: DeviceCreate):
    db_device = DeviceModel(**device.dict())
    db.add(db_device)
    db.commit()
    db.refresh(db_device)
    return db_device

def update_device(db: Session, device_id: int, device: DeviceBase):
    db_device = db.query(DeviceModel).filter(DeviceModel.device_id == device_id).first()
    if db_device:
        for key, value in device.dict(exclude_unset=True).items():
            setattr(db_device, key, value)
        db.commit()
        db.refresh(db_device)
    return db_device

def delete_device(db: Session, device_id: int):
    db_device = db.query(DeviceModel).filter(DeviceModel.device_id == device_id).first()
    if db_device:
        db.delete(db_device)
        db.commit()
    return db_device

# 设备使用记录
def get_device_usage_record(db: Session, record_id: int):
    return db.query(DeviceUsageRecordModel).filter(DeviceUsageRecordModel.record_id == record_id).first()

def get_device_usage_records(db: Session, skip: int = 0, limit: int = 100):
    return db.query(DeviceUsageRecordModel).offset(skip).limit(limit).all()

def create_device_usage_record(db: Session, record: DeviceUsageRecordCreate):
    db_record = DeviceUsageRecordModel(**record.dict())
    db.add(db_record)
    db.commit()
    db.refresh(db_record)
    return db_record

def update_device_usage_record(db: Session, record_id: int, record: DeviceUsageRecordBase):
    db_record = db.query(DeviceUsageRecordModel).filter(DeviceUsageRecordModel.record_id == record_id).first()
    if db_record:
        for key, value in record.dict(exclude_unset=True).items():
            setattr(db_record, key, value)
        db.commit()
        db.refresh(db_record)
    return db_record

def delete_device_usage_record(db: Session, record_id: int):
    db_record = db.query(DeviceUsageRecordModel).filter(DeviceUsageRecordModel.record_id == record_id).first()
    if db_record:
        db.delete(db_record)
        db.commit()
    return db_record

# 安防事件
def get_security_event(db: Session, event_id: int):
    return db.query(SecurityEventModel).filter(SecurityEventModel.event_id == event_id).first()

def get_security_events(db: Session, skip: int = 0, limit: int = 100):
    return db.query(SecurityEventModel).offset(skip).limit(limit).all()

def create_security_event(db: Session, event: SecurityEventCreate):
    db_event = SecurityEventModel(**event.dict())
    db.add(db_event)
    db.commit()
    db.refresh(db_event)
    return db_event

def update_security_event(db: Session, event_id: int, event: SecurityEventBase):
    db_event = db.query(SecurityEventModel).filter(SecurityEventModel.event_id == event_id).first()
    if db_event:
        for key, value in event.dict(exclude_unset=True).items():
            setattr(db_event, key, value)
        db.commit()
        db.refresh(db_event)
    return db_event

def delete_security_event(db: Session, event_id: int):
    db_event = db.query(SecurityEventModel).filter(SecurityEventModel.event_id == event_id).first()
    if db_event:
        db.delete(db_event)
        db.commit()
    return db_event

# 用户反馈
def get_user_feedback(db: Session, feedback_id: int):
    return db.query(UserFeedbackModel).filter(UserFeedbackModel.feedback_id == feedback_id).first()

def get_user_feedbacks(db: Session, skip: int = 0, limit: int = 100):
    return db.query(UserFeedbackModel).offset(skip).limit(limit).all()

def create_user_feedback(db: Session, feedback: UserFeedbackCreate):
    db_feedback = UserFeedbackModel(**feedback.dict())
    db.add(db_feedback)
    db.commit()
    db.refresh(db_feedback)
    return db_feedback

def update_user_feedback(db: Session, feedback_id: int, feedback: UserFeedbackBase):
    db_feedback = db.query(UserFeedbackModel).filter(UserFeedbackModel.feedback_id == feedback_id).first()
    if db_feedback:
        for key, value in feedback.dict(exclude_unset=True).items():
            setattr(db_feedback, key, value)
        db.commit()
        db.refresh(db_feedback)
    return db_feedback

def delete_user_feedback(db: Session, feedback_id: int):
    db_feedback = db.query(UserFeedbackModel).filter(UserFeedbackModel.feedback_id == feedback_id).first()
    if db_feedback:
        db.delete(db_feedback)
        db.commit()
    return db_feedback



#--------------------------------基础功能api路由 --------------------------------
Base.metadata.create_all(bind=engine)
app = FastAPI()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# 用户
@app.get("/")
def home():
    return {"message": "欢迎使用智能家居API"}

@app.post("/users/", response_model=UserResponse)
def create_user_api(user: UserCreate, db: Session = Depends(get_db)):
    try:
        return create_user(db, user)
    except Exception as e:
        raise HTTPException(400, detail=f"创建用户失败: {str(e)}")

@app.get("/users/", response_model=list[UserResponse])
def read_users(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    return get_users(db, skip, limit)

@app.get("/users/{user_id}", response_model=UserResponse)
def read_user(user_id: int, db: Session = Depends(get_db)):
    if user := get_user(db, user_id):
        return user
    raise HTTPException(404, detail=f"用户 {user_id} 不存在")

@app.put("/users/{user_id}", response_model=UserResponse)
def update_user_api(user_id: int, user: UserBase, db: Session = Depends(get_db)):
    if updated_user := update_user(db, user_id, user):
        return updated_user
    raise HTTPException(404, detail=f"用户 {user_id} 不存在")

@app.delete("/users/{user_id}")
def delete_user_api(user_id: int, db: Session = Depends(get_db)):
    if delete_user(db, user_id):
        return {"message": f"用户 {user_id} 已删除"}
    raise HTTPException(404, detail=f"用户 {user_id} 不存在")

# 设备信息
@app.post("/devices/", response_model=DeviceResponse)
def create_device_api(device: DeviceCreate, db: Session = Depends(get_db)):
    try:
        return create_device(db, device)
    except Exception as e:
        raise HTTPException(400, detail=f"创建设备失败: {str(e)}")

@app.get("/devices/", response_model=list[DeviceResponse])
def read_devices(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    return get_devices(db, skip, limit)

@app.get("/devices/{device_id}", response_model=DeviceResponse)
def read_device(device_id: int, db: Session = Depends(get_db)):
    if device := get_device(db, device_id):
        return device
    raise HTTPException(404, detail=f"设备 {device_id} 不存在")

# 设备使用记录
@app.get("/device_usage_records/{record_id}", response_model=DeviceUsageRecordResponse)
def get_device_usage_record_api(record_id: int, db: Session = Depends(get_db)):
    if record := get_device_usage_record(db, record_id):
        return record
    raise HTTPException(status_code=404, detail=f"使用记录 {record_id} 不存在")

@app.put("/device_usage_records/{record_id}", response_model=DeviceUsageRecordResponse)
def update_device_usage_record_api(
    record_id: int,
    record_data: DeviceUsageRecordBase,
    db: Session = Depends(get_db)
):
    if updated_record := update_device_usage_record(db, record_id, record_data):
        return updated_record
    raise HTTPException(status_code=404, detail=f"使用记录 {record_id} 不存在")

@app.delete("/device_usage_records/{record_id}")
def delete_device_usage_record_api(record_id: int, db: Session = Depends(get_db)):
    if delete_device_usage_record(db, record_id):
        return {"message": f"使用记录 {record_id} 已删除"}
    raise HTTPException(status_code=404, detail=f"使用记录 {record_id} 不存在")

# 安防事件
@app.get("/security_events/{event_id}", response_model=SecurityEventResponse)
def get_security_event_api(event_id: int, db: Session = Depends(get_db)):
    if event := get_security_event(db, event_id):
        return event
    raise HTTPException(status_code=404, detail=f"安防事件 {event_id} 不存在")

@app.put("/security_events/{event_id}", response_model=SecurityEventResponse)
def update_security_event_api(
    event_id: int,
    event_data: SecurityEventBase,
    db: Session = Depends(get_db)
):
    if updated_event := update_security_event(db, event_id, event_data):
        return updated_event
    raise HTTPException(status_code=404, detail=f"安防事件 {event_id} 不存在")

@app.delete("/security_events/{event_id}")
def delete_security_event_api(event_id: int, db: Session = Depends(get_db)):
    if delete_security_event(db, event_id):
        return {"message": f"安防事件 {event_id} 已删除"}
    raise HTTPException(status_code=404, detail=f"安防事件 {event_id} 不存在")

# 用户反馈
@app.get("/user_feedbacks/{feedback_id}", response_model=UserFeedbackResponse)
def get_user_feedback_api(feedback_id: int, db: Session = Depends(get_db)):
    if feedback := get_user_feedback(db, feedback_id):
        return feedback
    raise HTTPException(status_code=404, detail=f"用户反馈 {feedback_id} 不存在")

@app.put("/user_feedbacks/{feedback_id}", response_model=UserFeedbackResponse)
def update_user_feedback_api(
    feedback_id: int,
    feedback_data: UserFeedbackBase,
    db: Session = Depends(get_db)
):
    if updated_feedback := update_user_feedback(db, feedback_id, feedback_data):
        return updated_feedback
    raise HTTPException(status_code=404, detail=f"用户反馈 {feedback_id} 不存在")

@app.delete("/user_feedbacks/{feedback_id}")
def delete_user_feedback_api(feedback_id: int, db: Session = Depends(get_db)):
    if delete_user_feedback(db, feedback_id):
        return {"message": f"用户反馈 {feedback_id} 已删除"}
    raise HTTPException(status_code=404, detail=f"用户反馈 {feedback_id} 不存在")




# -------------------------- 三个小问题api路由 --------------------------
# 分析不同设备的使用频率和使用时间段
@app.get("/analysis/device_usage_frequency")
async def analyze_device_usage(start_date: str = '2023-04-01', end_date: str = '2023-04-06'):
    """
    分析设备使用频率和时间分布
    返回格式：
    {
        "summary": [
            {
                "device_id": 1,
                "device_name": "客厅空调",
                "usage_count": 15,
                "avg_duration_hours": 2.5,
                "hourly_distribution": {
                    "8": 3,
                    "9": 5,
                    ...
                }
            },
            ...
        ],
        "heatmap_image": "base64str"
    }
    """
    # 获取基础数据
    query = text("""
                 SELECT d.device_id,
                        d.name,
                        dur.start_time,
                        dur.end_time,
                        EXTRACT(HOUR FROM dur.start_time)                          as start_hour,
                        EXTRACT(EPOCH FROM (dur.end_time - dur.start_time)) / 3600 as duration_hours
                 FROM device_usage_records dur
                          JOIN devices d ON dur.device_id = d.device_id
                 WHERE dur.start_time BETWEEN :start_date AND :end_date
                 """)

    with SessionLocal() as db:
        result = db.execute(query, {'start_date': start_date, 'end_date': end_date})
        data = [dict(row) for row in result.mappings()]

    df = pd.DataFrame(data)

    # 分析统计
    analysis = []
    for device_id, group in df.groupby('device_id'):
        hourly_dist = group['start_hour'].value_counts().to_dict()
        analysis.append({
            "device_id": device_id,
            "device_name": group.iloc[0]['name'],
            "usage_count": len(group),
            "avg_duration_hours": round(group['duration_hours'].mean(), 1),
            "hourly_distribution": {str(int(k)): int(v) for k, v in hourly_dist.items()}
        })


    return JSONResponse({
        "summary": analysis
    })

# 找出用户的使用习惯（如哪些设备经常同时使用）
@app.get("/analysis/co_usage_patterns")
async def analyze_co_usage_patterns(time_window: int = 15):
    """
    分析设备同时使用模式
    返回格式：
    {
        "co_usage": [
            {
                "device_pair": ["空调", "智能灯"],
                "co_occurrence_count": 23,
                "users": [1,5,7]
            },
            ...
        ]
    }
    """
    query = text("""
                 SELECT user_id, device_id, start_time, end_time
                 FROM device_usage_records
                 ORDER BY user_id, start_time
                 """)

    with SessionLocal() as db:
        result = db.execute(query)
        data = [dict(row) for row in result.mappings()]

    # 按用户分组分析
    co_occurrence = defaultdict(int)
    pair_users = defaultdict(set)

    for user_id, user_records in pd.DataFrame(data).groupby('user_id'):
        records = user_records.sort_values('start_time').to_dict('records')

        # 创建时间窗口
        time_groups = []
        current_group = []
        prev_end = None

        for record in records:
            if prev_end is None or record['start_time'] > prev_end + pd.Timedelta(minutes=time_window):
                if current_group:
                    time_groups.append(current_group)
                current_group = [record]
            else:
                current_group.append(record)
            prev_end = max(prev_end or record['end_time'], record['end_time'])

        if current_group:
            time_groups.append(current_group)

        # 统计每个时间窗口内的设备组合
        for group in time_groups:
            devices = list({r['device_id'] for r in group})
            for pair in combinations(sorted(devices), 2):
                co_occurrence[pair] += 1
                pair_users[pair].add(user_id)

    # 获取设备名称映射
    with SessionLocal() as db:
        devices = db.query(DeviceModel.device_id, DeviceModel.name).all()
    device_names = {d.device_id: d.name for d in devices}

    # 生成结果
    result = []
    for (d1, d2), count in sorted(co_occurrence.items(), key=lambda x: -x[1])[:10]:
        result.append({
            "device_pair": [device_names[d1], device_names[d2]],
            "co_occurrence_count": count,
            "users": list(pair_users[(d1, d2)])[:5]  # 显示前5个用户
        })

    return JSONResponse({"co_usage": result})

# 分析房屋面积对设备使用行为的影响
@app.get("/analysis/house_area_impact")
async def analyze_house_area_impact():
    """
    分析房屋面积对设备使用的影响
    返回格式：
    {
        "correlation": {
            "energy_consumption": 0.78,
            "usage_count": 0.65
        },
        "area_groups": [
            {
                "area_range": "0-80",
                "avg_energy": 15.2,
                "avg_duration": 2.3,
                "top_devices": ["空调", "热水器"]
            },
            ...
        ]
    }
    """
    query = text("""
                 SELECT u.user_id,
                        u.house_area,
                        d.device_id,
                        d.name                                                          as device_name,
                        COUNT(dur.record_id)                                            as usage_count,
                        AVG(EXTRACT(EPOCH FROM (dur.end_time - dur.start_time)) / 3600) as avg_duration,
                        SUM(dur.energy_consumption)                                     as total_energy
                 FROM users u
                          JOIN device_usage_records dur ON u.user_id = dur.user_id
                          JOIN devices d ON dur.device_id = d.device_id
                 GROUP BY u.user_id, d.device_id, d.name
                 """)

    with SessionLocal() as db:
        result = db.execute(query)
        data = [dict(row) for row in result.mappings()]

    df = pd.DataFrame(data)

    # 计算相关系数
    corr_energy = pearsonr(df['house_area'], df['total_energy'])[0]
    corr_count = pearsonr(df.groupby('user_id')['usage_count'].sum(),
                          df.groupby('user_id')['house_area'].first())[0]

    # 按面积分组分析
    df['area_group'] = pd.cut(df['house_area'],
                              bins=[0, 80, 120, 200],
                              labels=["0-80", "81-120", "121-200"])

    area_analysis = []
    for group, sub_df in df.groupby('area_group'):
        top_devices = sub_df.groupby('device_name')['usage_count'] \
            .sum().nlargest(3).index.tolist()

        area_analysis.append({
            "area_range": group,
            "avg_energy": round(sub_df['total_energy'].mean(), 1),
            "avg_duration": round(sub_df['avg_duration'].mean(), 1),
            "top_devices": top_devices
        })

    return JSONResponse({
        "correlation": {
            "energy_consumption": round(corr_energy, 2),
            "usage_count": round(corr_count, 2)
        },
        "area_groups": area_analysis
    })



#---------------------------------子问题----------------------------------
# 设备能耗排名分析
@app.get("/analysis/device_energy_ranking")
async def device_energy_ranking(
        start_date: date = date(2023, 4, 1),
        end_date: date = date(2023, 4, 6),
        db: Session = Depends(get_db)
):
    """
    获取设备能耗排名
    参数:
        start_date: 开始日期 (默认: 2023-04-01)
        end_date: 结束日期 (默认: 2023-04-06)
    返回:
        {
            "summary": [
                {
                    "device_id": 1,
                    "name": "空调",
                    "type": "air_conditioner",
                    "total_energy": 150.5
                },
                ...
            ]
        }
    """
    query = text("""
        SELECT 
            d.device_id,
            d.name,
            d.type,
            SUM(dur.energy_consumption) as total_energy
        FROM device_usage_records dur
        JOIN devices d ON dur.device_id = d.device_id
        WHERE dur.start_time BETWEEN :start_date AND :end_date
        GROUP BY d.device_id, d.name, d.type
        ORDER BY total_energy DESC
        LIMIT 10
    """)

    result = db.execute(query, {"start_date": start_date, "end_date": end_date})
    return {"summary": [dict(row) for row in result.mappings()]}

# 面积与能耗分析
@app.get("/analysis/area_vs_energy", summary="Area Vs Energy")
async def area_vs_energy(
        group_type: str = Query("auto", enum=["auto", "custom"], description="分组类型：auto(自动)/custom(自定义)"),
        custom_ranges: str = Query("65-85,86-120,121-150", description="自定义分组格式: min1-max1,min2-max2"),
        db: Session = Depends(get_db)
):
    """
    分析房屋面积与设备能耗的关系（基于用户表格数据优化）
    参数:
      - group_type: 分组方式
      - custom_ranges: 当group_type=custom时生效
    返回格式:
    {
        "area_groups": [
            {
                "area_range": "65-85",
                "avg_energy": 45.2,
                "top_devices": ["空调", "热水器"],
                "user_count": 7
            },
            ...
        ],
        "correlation": 0.78
    }
    """
    # 获取基础数据
    query = text("""
        SELECT 
            u.house_area,
            d.type,
            AVG(dur.energy_consumption) as avg_energy,
            COUNT(*) as usage_count
        FROM users u
        JOIN device_usage_records dur ON u.user_id = dur.user_id
        JOIN devices d ON dur.device_id = d.device_id
        GROUP BY u.house_area, d.type
    """)
    result = db.execute(query)
    df = pd.DataFrame([dict(row) for row in result.mappings()])

    # 动态分组逻辑
    if group_type == "custom":
        bins = []
        for range_str in custom_ranges.split(','):
            min_val, max_val = map(float, range_str.split('-'))
            bins.extend([min_val, max_val])
        bins = sorted(set(bins))
    else:
        # 根据用户表格数据自动分组（65-150㎡范围）
        bins = [65, 85, 120, 150]  # 基于数据分布的分组

    labels = [f"{bins[i]}-{bins[i + 1]}" for i in range(len(bins) - 1)]
    df['area_range'] = pd.cut(df['house_area'], bins=bins, labels=labels, include_lowest=True)

    # 计算相关系数
    corr = round(df['house_area'].corr(df['avg_energy']), 2)

    # 生成分组结果
    area_groups = []
    for group, sub_df in df.groupby('area_range', observed=True):
        user_count = len(sub_df['house_area'].unique())  # 该分组的用户数
        top_devices = sub_df.nlargest(3, 'usage_count')['type'].tolist()
        area_groups.append({
            "area_range": str(group),
            "avg_energy": round(sub_df['avg_energy'].mean(), 2),
            "top_devices": top_devices,
            "user_count": user_count
        })

    return {
        "area_groups": area_groups,
        "correlation": corr,
        "data_range": {"min": df['house_area'].min(), "max": df['house_area'].max()}
    }


# 温度与功率分析
@app.get("/analysis/temperature_vs_power", summary="Temperature Vs Power")
async def temperature_vs_power(
        temp_min: float = Query(15, ge=10, le=40, description="最低温度(℃)"),
        temp_max: float = Query(30, ge=10, le=40, description="最高温度(℃)"),
        db: Session = Depends(get_db)
):
    """
    温度与设备功率关系分析（基于瞬时功率计算）
    """
    try:
        # 获取数据（包含时间字段）
        query = text("""
            SELECT 
                temperature, 
                energy_consumption,
                start_time,
                end_time
            FROM device_usage_records
            WHERE temperature BETWEEN :min AND :max
              AND temperature IS NOT NULL
              AND start_time IS NOT NULL
              AND end_time IS NOT NULL
            ORDER BY temperature
        """)
        result = db.execute(query, {"min": temp_min, "max": temp_max})
        records = [dict(row) for row in result.mappings()]

        if len(records) < 3:
            return {
                "data": [],
                "optimal_temp": None,
                "warning": "至少需要3条有效数据",
                "unit": "kW"
            }

        # 计算功率（kWh → kW）
        df = pd.DataFrame(records)

        # 计算持续时间（小时）
        df['duration_hours'] = df.apply(
            lambda x: (x['end_time'] - x['start_time']).total_seconds() / 3600,
            axis=1
        )

        # 过滤无效持续时间
        df = df[df['duration_hours'] > 0]

        if len(df) < 3:
            return {
                "data": [],
                "optimal_temp": None,
                "warning": "有效数据不足（需持续时间>0）",
                "unit": "kW"
            }

        # 计算功率（kW）
        df['power_kw'] = df['energy_consumption'] / df['duration_hours']

        # 温度分箱分析
        bins = np.arange(max(15, temp_min), min(33, temp_max + 1), 3)
        df['temp_bin'] = pd.cut(df['temperature'], bins=bins)

        if df['temp_bin'].isnull().all():
            return {
                "data": [],
                "optimal_temp": None,
                "error": "无有效温度分箱",
                "unit": "kW"
            }

        # 计算最佳温度（功率最低点）
        grouped = df.groupby('temp_bin')['power_kw'].mean()
        optimal_bin = grouped.idxmin()
        optimal_temp = round(optimal_bin.mid, 1)

        return {
            "data": df[['temperature', 'power_kw']].to_dict('records'),
            "optimal_temp": optimal_temp,
            "stats": {
                "samples": len(df),
                "temp_range": [temp_min, temp_max],
                "avg_power": df['power_kw'].mean(

                ),
                "min_power": df['power_kw'].min(),
                "max_power": df['power_kw'].max()
            },
            "unit": "kW"
        }

    except Exception as e:
        raise HTTPException(500, detail=f"分析失败: {str(e)}")

# 不同房间能耗分析
@app.get("/advanced_analysis/energy_breakdown")
def get_energy_breakdown(db: Session = Depends(get_db)):
    """获取房间-设备类型-能耗三维数据"""
    result = db.query(
        DeviceModel.location.label("room"),
        DeviceModel.type.label("device_type"),
        func.sum(DeviceUsageRecordModel.energy_consumption).label("total_energy")
    ).join(DeviceUsageRecordModel).group_by("room", "device_type").all()

    return [
        {"room": r.room,
         "device_type": r.device_type,
         "total_energy": float(r.total_energy)}
        for r in result
    ]

# 用户反馈分析
@app.get("/advanced_analysis/sentiment_details")
def get_sentiment_details(db: Session = Depends(get_db)):
    """获取所有反馈文本（不区分设备）"""
    feedbacks = db.query(UserFeedbackModel.feedback_text).filter(
        UserFeedbackModel.feedback_text.isnot(None)
    ).all()
    return [f[0] for f in feedbacks]

# 用户满意度分析
@app.get("/advanced_analysis/device_feedback_details")
def get_device_feedback_details(db: Session = Depends(get_db)):
    """获取评分次数超过2次的设备反馈详情"""
    # 子查询：筛选总评分>2的设备
    subquery = (
        db.query(
            cast(UserFeedbackModel.device_id, Integer).label("device_id")
        )
        .group_by("device_id")
        .having(func.count() > 2)
        .subquery()
    )

    # 主查询：三表正确关联
    result = (
        db.query(
            DeviceModel.name.label("device_name"),
            UserFeedbackModel.user_id,
            UserFeedbackModel.feedback_date,
            UserFeedbackModel.satisfaction_rating
        )
        # 先关联反馈表与子查询
        .join(
            subquery,
            cast(UserFeedbackModel.device_id, Integer) == subquery.c.device_id
        )
        # 再关联设备表
        .join(
            DeviceModel,
            DeviceModel.device_id == subquery.c.device_id
        )
        .order_by(desc(UserFeedbackModel.feedback_date))
        .all()
    )

    return [
        {
            "device_name": row.device_name,
            "user_id": row.user_id,
            "feedback_time": str(row.feedback_date),
            "rating": row.satisfaction_rating
        }
        for row in result
    ]



