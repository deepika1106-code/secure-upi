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
            avg = d.execute("SELECT AVG(amount) FROM txns WHERE user_id=? AND decision='APPROVED'", (uid,)).fetchone()[0]
            ratio = min(amount / avg, 10) if avg else 1.0
            cutoff = (now_ist() - timedelta(hours=1)).strftime("%Y-%m-%d %H:%M")
            tlh = d.execute("SELECT COUNT(*) FROM txns WHERE user_id=? AND created>=?", (uid, cutoff)).fetchone()[0]
            dev = hashlib.md5(request.user_agent.string.encode()).hexdigest()[:10]
            last = d.execute("SELECT device FROM txns WHERE user_id=? ORDER BY id DESC LIMIT 1", (uid,)).fetchone()
            changed = int(bool(last) and last["device"] != dev)
            lat, lon = num(f.get("lat")), num(f.get("lon"))
            prev = d.execute("SELECT lat, lon FROM txns WHERE user_id=? AND lat IS NOT NULL ORDER BY id DESC LIMIT 1", (uid,)).fetchone()
            dist = km(prev["lat"], prev["lon"], lat, lon) if (prev and lat is not None) else 0.0
            if f.get("demo_dist"): dist = float(f["demo_dist"])
            row = dict(amount=amount, hour=hour, receiver_new=int(seen == 0), receiver_flags=flags, kw_hits=kw,
                       digit_ratio=dr, unknown_handle=unk, txns_last_hour=tlh, amount_ratio=round(ratio, 2),
                       device_changed=changed, distance_km=round(dist, 1))
            p = float(model.predict_proba(pd.DataFrame([row])[FEATURES])[0][1])
            dec = decide(p)
            why = []
            if hour < 5: why.append("Late-night payment (%d:00)" % hour)
            if kw: why.append("ID looks suspicious (scam words or fake-looking pattern)")
            if amount > 100000: why.append("Amount is above the normal UPI limit of Rs 1,00,000")
            if flags: why.append("This receiver was blocked %d time(s) before" % flags)
            if seen == 0: why.append("First payment to this receiver")
            if unk: why.append("Unrecognised bank handle")
            if dr > 0.3: why.append("ID is mostly random digits")
            if ratio > 3: why.append("Amount is %dx your usual" % ratio)
            if amount >= 25000: why.append("High amount")
            if tlh >= 3: why.append("%d payments in the last hour" % tlh)
            if changed: why.append("Payment from a new device")
            if dist > 100: why.append("Location %d km from your usual area" % dist)
            if not why: why = ["No risk signals found"]
            d.execute("INSERT INTO txns(user_id,receiver,amount,hour,risk,decision,created,device,lat,lon,reasons) "
                      "VALUES(?,?,?,?,?,?,?,?,?,?,?)", (uid, rcv, amount, hour, p, dec,
                      now_ist().strftime("%Y-%m-%d %H:%M"), dev, lat, lon, " | ".join(why)))
            d.commit()
            result = dict(risk=round(p*100, 1), decision=dec, receiver=rcv, amount=amount, reasons=why,
                          sim=bool(f.get("demo_hour") or f.get("demo_dist")))
    q = "SELECT * FROM txns" + ("" if session["role"] == "admin" else " WHERE user_id=%d" % session["uid"])
    rows = db().execute(q + " ORDER BY id DESC LIMIT 10").fetchall()
    return render_template("index.html", result=result, rows=rows, now=now_ist().strftime("%I:%M %p"))

@app.route("/dashboard")
@login_required
def dashboard():
    if session["role"] != "admin": return redirect("/")
    return render_template("dashboard.html")

@app.route("/api/stats")
@login_required
def stats():
    if session["role"] != "admin": return jsonify(error="forbidden"), 403
    c = db().execute("SELECT decision, COUNT(*) n FROM txns GROUP BY decision").fetchall()
    h = db().execute("SELECT hour, COUNT(*) n FROM txns WHERE decision!='APPROVED' GROUP BY hour ORDER BY hour").fetchall()
    t = db().execute("SELECT COUNT(*) n, COALESCE(SUM(CASE WHEN decision='BLOCKED' THEN amount END),0) saved FROM txns").fetchone()
    return jsonify(decisions={r["decision"]: r["n"] for r in c}, by_hour={r["hour"]: r["n"] for r in h},
                   total=t["n"], saved=t["saved"])

# ---------- PWA (makes the website installable as a phone app) ----------
import zlib, struct
from functools import lru_cache
from flask import Response

@lru_cache(maxsize=4)
def make_icon(s):
    def seg(px, py, ax, ay, bx, by):
        dx, dy = bx-ax, by-ay
        t = max(0, min(1, ((px-ax)*dx + (py-ay)*dy) / (dx*dx + dy*dy)))
        return ((px-ax-t*dx)**2 + (py-ay-t*dy)**2) ** 0.5
    raw = bytearray()
    for y in range(s):
        raw.append(0)
        v = y / s
        for x in range(s):
            u = x / s
            k = (u + v) / 2
            r, g, b = int(99 + (236-99)*k), int(102 + (72-102)*k), int(241 + (153-241)*k)
            w = abs(u - 0.5)
            if 0.24 <= v <= 0.52: shield = w <= 0.22
            elif 0.52 < v <= 0.78: shield = w <= 0.22 * (1 - (v-0.52)/0.26) ** 0.6
            else: shield = False
            if shield:
                r, g, b = 255, 255, 255
                if min(seg(u, v, 0.40, 0.50, 0.47, 0.57), seg(u, v, 0.47, 0.57, 0.61, 0.40)) < 0.035:
                    r, g, b = 99, 102, 241
            raw += bytes((r, g, b))
    def chunk(t, d):
        c = struct.pack(">I", len(d)) + t + d
        return c + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", s, s, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(bytes(raw), 9)) + chunk(b"IEND", b""))

@app.route("/icon-<int:size>.png")
def icon(size):
    size = size if size in (180, 192, 512) else 192
    return Response(make_icon(size), mimetype="image/png", headers={"Cache-Control": "public, max-age=86400"})

@app.route("/manifest.json")
def manifest():
    return jsonify(name="Secure UPI Fraud Shield", short_name="SecureUPI", description="ML-driven UPI fraud detection system",
                   start_url="/", scope="/", display="standalone", orientation="portrait",
                   background_color="#312e81", theme_color="#6d28d9",
                   icons=[{"src": "/icon-192.png", "sizes": "192x192", "type": "image/png", "purpose": "any maskable"},
                          {"src": "/icon-512.png", "sizes": "512x512", "type": "image/png", "purpose": "any maskable"}])

SW_JS = """
self.addEventListener('install', e => self.skipWaiting());
self.addEventListener('activate', e => e.waitUntil(self.clients.claim()));
self.addEventListener('fetch', e => {
  if (e.request.mode === 'navigate') {
    e.respondWith(fetch(e.request).catch(() => new Response(
      '<body style="font-family:sans-serif;text-align:center;padding:60px;background:#312e81;color:#fff">' +
      '<h2>You are offline</h2><p>Secure UPI needs internet to check payments. Please reconnect.</p></body>',
      {headers: {'Content-Type': 'text/html'}})));
  }
});
"""

@app.route("/sw.js")
def service_worker():
    return Response(SW_JS, mimetype="application/javascript", headers={"Cache-Control": "no-cache"})

if __name__ == "__main__":
    app.run(debug=True)
