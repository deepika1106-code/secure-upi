import re, math, hashlib, sqlite3, joblib, pandas as pd
from datetime import datetime, timedelta, timezone
from flask import Flask, render_template, request, redirect, session, jsonify, g, flash
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = "change-this-secret-in-production"
model = joblib.load("model.pkl")
FEATURES = ["amount", "hour", "receiver_new", "receiver_flags", "kw_hits", "digit_ratio",
            "unknown_handle", "txns_last_hour", "amount_ratio", "device_changed", "distance_km"]
IST = timezone(timedelta(hours=5, minutes=30))
HANDLES = {"upi", "ybl", "ibl", "axl", "okaxis", "oksbi", "okhdfcbank", "okicici", "paytm", "apl", "sbi",
           "hdfcbank", "icici", "axisbank", "pnb", "boi", "cnrb", "kotak", "idfcbank", "indus", "federal",
           "yesbank", "postbank", "aubank", "rbl", "bandhan", "unionbank", "jio", "airtel", "freecharge", "slice"}
KEYWORDS = ["refund", "kyc", "lottery", "reward", "cashback", "prize", "offer", "claim", "support", "helpdesk",
            "care", "verify", "bonus", "gift", "winner", "loan", "urgent", "reversal", "customer"]
VALID = re.compile(r"[6-9]\d{9}|[\w.\-]{2,64}@[a-z]{2,32}")

def now_ist(): return datetime.now(IST)

def db():
    if "db" not in g:
        g.db = sqlite3.connect("upi.db"); g.db.row_factory = sqlite3.Row
    return g.db

@app.teardown_appcontext
def close(e):
    d = g.pop("db", None)
    if d: d.close()

def init_db():
    c = sqlite3.connect("upi.db")
    c.executescript("""CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY, username TEXT UNIQUE,
        password TEXT, role TEXT);
      CREATE TABLE IF NOT EXISTS txns(id INTEGER PRIMARY KEY, user_id INTEGER, receiver TEXT, amount REAL,
        hour INT, risk REAL, decision TEXT, created TEXT, device TEXT, lat REAL, lon REAL, reasons TEXT);""")
    c.commit(); c.close()
init_db()

def login_required(f):
    from functools import wraps
    @wraps(f)
    def w(*a, **k):
        if "uid" not in session: return redirect("/login")
        return f(*a, **k)
    return w

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        u, p = request.form["username"].strip(), request.form["password"]
        if not u or len(p) < 6:
            flash("Username required, password min 6 chars"); return render_template("auth.html", mode="Register")
        role = "admin" if db().execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0 else "user"
        try:
            db().execute("INSERT INTO users(username,password,role) VALUES(?,?,?)", (u, generate_password_hash(p), role))
            db().commit(); return redirect("/login")
        except sqlite3.IntegrityError:
            flash("Username already taken")
    return render_template("auth.html", mode="Register")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        r = db().execute("SELECT * FROM users WHERE username=?", (request.form["username"],)).fetchone()
        if r and check_password_hash(r["password"], request.form["password"]):
            session.update(uid=r["id"], name=r["username"], role=r["role"]); return redirect("/")
        flash("Invalid credentials")
    return render_template("auth.html", mode="Login")

@app.route("/logout")
def logout():
    session.clear(); return redirect("/login")

def decide(p):
    return "APPROVED" if p < 0.25 else ("OTP RE-VERIFY" if p < 0.55 else "BLOCKED")

KEYWORDS += ["spam", "fraud", "scam", "hack", "fake", "free", "earn", "invest", "crypto", "casino", "betting",
             "rummy", "money", "lucky", "jackpot", "trading", "paisa"]

def analyze_id(rcv):
    local, _, handle = rcv.partition("@")
    phone = bool(re.fullmatch(r"[6-9]\d{9}", local))
    kw = min(sum(k in local for k in KEYWORDS), 3)
    if phone and (len(set(local)) <= 3 or local in "01234567890123456789" or local in "98765432109876543210"):
        kw = max(kw, 2)   # fake-looking phone number such as 9999999999
    dr = 0.0 if phone else round(sum(ch.isdigit() for ch in local) / max(len(local), 1), 2)
    unk = int(bool(handle) and handle not in HANDLES)
    return kw, dr, unk

class HybridModel:
    """Machine Learning probability + safety rules (hybrid fraud engine)."""
    def __init__(self, ml): self.ml = ml
    def predict_proba(self, X):
        r = X.iloc[0]
        floor = 0.0
        if r.amount > 100000: floor = 0.90                         # above the normal UPI limit
        if r.receiver_flags >= 2: floor = max(floor, 0.85)         # repeatedly blocked receiver
        if r.kw_hits >= 2 or (r.kw_hits >= 1 and r.unknown_handle): floor = max(floor, 0.50)
        X = X.copy(); X["amount"] = X["amount"].clip(upper=100000)
        p = max(float(self.ml.predict_proba(X)[0][1]), floor)
        return [[1 - p, p]]

model = HybridModel(model)

def km(a, b, c, d):
    p = math.pi / 180
    x = math.sin((c-a)*p/2)**2 + math.cos(a*p)*math.cos(c*p)*math.sin((d-b)*p/2)**2
    return 12742 * math.asin(math.sqrt(x))

def num(v):
    try: return float(v)
    except (TypeError, ValueError): return None

@app.route("/", methods=["GET", "POST"])
@login_required
def index():
    result = None
    if request.method == "POST":
        f, uid, d = request.form, session["uid"], db()
        rcv = f["receiver"].strip().lower().replace(" ", "")
        amount = num(f["amount"])
        if not VALID.fullmatch(rcv) or not amount or amount <= 0:
            flash("Enter a valid UPI ID (name@bank) or 10-digit mobile number, and an amount")
        else:
            kw, dr, unk = analyze_id(rcv)
            hour = int(num(f.get("demo_hour")) if f.get("demo_hour") else now_ist().hour)
            seen = d.execute("SELECT COUNT(*) FROM txns WHERE user_id=? AND receiver=? AND decision='APPROVED'", (uid, rcv)).fetchone()[0]
            flags = min(d.execute("SELECT COUNT(*) FROM txns WHERE receiver=? AND decision='BLOCKED'", (rcv,)).fetchone()[0], 5)
            avg = d.execute("SELECT AVG(amount) FROM txns WHERE user_id=? AND decision='APPROVED'",
