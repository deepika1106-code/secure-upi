import sqlite3, joblib, pandas as pd
from datetime import datetime
from flask import Flask, render_template, request, redirect, session, jsonify, g, flash
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = "change-this-secret-in-production"
model = joblib.load("model.pkl")
FEATURES = ["amount", "hour", "new_beneficiary", "device_changed", "distance_km", "txns_last_hour"]

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
      CREATE TABLE IF NOT EXISTS txns(id INTEGER PRIMARY KEY, user_id INTEGER, upi_id TEXT, amount REAL,
        hour INT, new_beneficiary INT, device_changed INT, distance_km REAL, txns_last_hour INT,
        risk REAL, decision TEXT, created TEXT);""")
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
    return "APPROVED" if p < 0.30 else ("OTP RE-VERIFY" if p < 0.70 else "BLOCKED")

@app.route("/", methods=["GET", "POST"])
@login_required
def index():
    result = None
    if request.method == "POST":
        f = request.form
        row = dict(amount=float(f["amount"]), hour=int(f["hour"]), new_beneficiary=int("new_beneficiary" in f),
                   device_changed=int("device_changed" in f), distance_km=float(f["distance_km"]),
                   txns_last_hour=int(f["txns_last_hour"]))
        p = float(model.predict_proba(pd.DataFrame([row])[FEATURES])[0][1])
        d = decide(p)
        db().execute("INSERT INTO txns(user_id,upi_id,amount,hour,new_beneficiary,device_changed,distance_km,"
                     "txns_last_hour,risk,decision,created) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                     (session["uid"], f["upi_id"], row["amount"], row["hour"], row["new_beneficiary"],
                      row["device_changed"], row["distance_km"], row["txns_last_hour"], p, d,
                      datetime.now().strftime("%Y-%m-%d %H:%M")))
        db().commit(); result = dict(risk=round(p*100, 1), decision=d, upi_id=f["upi_id"], amount=row["amount"])
    q = "SELECT * FROM txns" + ("" if session["role"] == "admin" else " WHERE user_id=%d" % session["uid"])
    rows = db().execute(q + " ORDER BY id DESC LIMIT 10").fetchall()
    return render_template("index.html", result=result, rows=rows, now_hour=datetime.now().hour)

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
