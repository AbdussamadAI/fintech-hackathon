"""Synthetic Nigerian bank-complaint stream with injected incidents (ground truth for ComplaintPulse).

Writes data/<name>/:
  complaints.parquet  one row per complaint: ts, channel, language, region, product, issue (gold), text,
                      metadata a bank already holds (app_version, device_os, switch, telco, atm_id, card_scheme,
                      segment), amount, incident_id (-1 = background), repost_of (-1 = original)
  events.parquet      operational events (releases, switch degradation, maintenance), causal and decoy
  incidents.json      injected incidents: start, peak, end, issue(s), cause attributes, linked event
All brand-like entities are generic ("Switch C", "Telco 2", "Card scheme B") so the demo never implies that a
real company had an outage.

Usage: python generator/generate.py --days 30 --per-day 8000 --seed 7
"""
import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

HOUR, DAY = 3600, 86400

# ------------------------------------------------------------------------------------------- geography
REGIONS = {  # state: (weight, zone)
    "Lagos": (24, "SW"), "FCT": (8, "NC"), "Kano": (7, "NW"), "Rivers": (6, "SS"), "Oyo": (5, "SW"),
    "Kaduna": (4, "NW"), "Ogun": (4, "SW"), "Enugu": (3, "SE"), "Anambra": (3, "SE"), "Delta": (3, "SS"),
    "Edo": (3, "SS"), "Imo": (2.5, "SE"), "Abia": (2, "SE"), "Plateau": (2, "NC"), "Kwara": (2, "NC"),
    "Osun": (2, "SW"), "Ondo": (1.8, "SW"), "Ekiti": (1.2, "SW"), "Akwa Ibom": (2, "SS"), "Cross River": (1.5, "SS"),
    "Bauchi": (1.5, "NE"), "Borno": (1.5, "NE"), "Sokoto": (1.5, "NW"), "Katsina": (1.8, "NW"), "Niger": (1.5, "NC"),
    "Benue": (1.5, "NC"), "Kogi": (1.3, "NC"), "Adamawa": (1.2, "NE"), "Gombe": (1, "NE"), "Jigawa": (1, "NW"),
    "Kebbi": (1, "NW"), "Zamfara": (1, "NW"), "Taraba": (0.8, "NE"), "Yobe": (0.8, "NE"), "Nasarawa": (1, "NC"),
    "Ebonyi": (1, "SE"), "Bayelsa": (1, "SS"),
}
ZONE_LANG = {  # language mix by geopolitical zone (complaints, not population)
    "SW": {"English": .45, "Pidgin": .25, "Yoruba": .30},
    "SE": {"English": .45, "Pidgin": .25, "Igbo": .30},
    "SS": {"English": .45, "Pidgin": .50, "Igbo": .05},
    "NW": {"English": .30, "Pidgin": .10, "Hausa": .60},
    "NE": {"English": .35, "Pidgin": .10, "Hausa": .55},
    "NC": {"English": .50, "Pidgin": .20, "Hausa": .25, "Yoruba": .05},
}
CHANNELS = {"call_centre": .30, "app_chat": .20, "social": .18, "email": .10, "app_review": .10, "branch": .12}
CHANNEL_ENGLISH_BIAS = {"email": .6, "app_review": .3, "branch": .2, "call_centre": 0, "app_chat": .1, "social": 0}

# ------------------------------------------------------------------------------------------- issues
# issue: (product, base share, needs amount)
ISSUES = {
    "transfer_debited_not_received": ("transfers", 14, True),
    "reversal_delayed": ("transfers", 8, True),
    "wrong_beneficiary": ("transfers", 2, True),
    "ussd_session_timeout": ("ussd", 6, False),
    "ussd_wrong_charges": ("ussd", 2, True),
    "atm_debited_no_cash": ("cards", 7, True),
    "card_declined": ("cards", 5, False),
    "card_blocked": ("cards", 2, False),
    "pos_debited_no_value": ("pos", 5, True),
    "app_login_failure": ("mobile_app", 6, False),
    "otp_not_received": ("mobile_app", 5, False),
    "app_crash": ("mobile_app", 3, False),
    "bvn_nin_update": ("account", 4, False),
    "account_restricted": ("account", 4, False),
    "unauthorised_debit": ("account", 3, True),
    "sms_charges": ("charges", 3, True),
    "maintenance_fee": ("charges", 2, True),
    "loan_deduction": ("loans", 2, True),
    "long_wait_time": ("service", 4, False),
    "staff_attitude": ("service", 2, False),
}
APP_VERSIONS = ["4.0.9", "4.1.0", "4.1.2", "4.1.5", "4.2.0", "4.2.1", "4.2.3"]
SWITCHES = ["Switch A", "Switch B", "Switch C", "Switch D", "Switch E"]
TELCOS = ["Telco 1", "Telco 2", "Telco 3", "Telco 4"]
CARD_SCHEMES = ["Card scheme A", "Card scheme B", "Card scheme C"]
SEGMENTS = {"retail": .62, "student": .14, "trader": .12, "pensioner": .05, "sme": .07}

# ------------------------------------------------------------------------------------------- text
# Compositional templates per language: opener + issue clause + (amount) + demand. Short, colloquial, varied.
OPEN = {
    "English": ["Good day,", "Hello,", "Please help,", "I am very disappointed.", "This is unacceptable.", "", "Hi,"],
    "Pidgin": ["Abeg,", "Una good morning o,", "Wetin dey happen?", "Na wa for una o.", "Abeg una help me,", "", "Ehn,"],
    "Hausa": ["Don Allah,", "Sannu,", "Ina kwana,", "Wannan ba daidai ba ne.", "Don Allah ku taimaka,", ""],
    "Yoruba": ["E jowo,", "E kaaro,", "E ba mi,", "Eleyi ko dara rara.", "E jowo e ran mi lowo,", ""],
    "Igbo": ["Biko,", "Ndewo,", "Unu, biko nyere m aka,", "Nke a adighi mma.", "Biko,", ""],
}
DEMAND = {
    "English": ["Please resolve urgently.", "I need my money back.", "Kindly look into it.", "Fix this now!", "I have complained twice already.", ""],
    "Pidgin": ["Abeg make una sort am quick quick.", "I want my money back o.", "Una no dey try at all.", "Make una fix am.", ""],
    "Hausa": ["Ina son a mayar min da kudina.", "Ku gyara wannan da sauri.", "Na kai kara sau biyu.", ""],
    "Yoruba": ["Mo fe owo mi pada.", "E tete yanju re.", "Mo ti so tele.", ""],
    "Igbo": ["Achoro m ego m.", "Biko dozie ya ngwa ngwa.", "Agwala m unu ugboro abuo.", ""],
}
CLAUSE = {  # {amt} filled for money issues
    "transfer_debited_not_received": {
        "English": ["I transferred {amt} and my account was debited but the beneficiary did not receive it.", "Transfer of {amt} debited since morning, recipient has not received."],
        "Pidgin": ["I send {amt} since morning, una don debit me but e no reach.", "Transfer of {amt} no land, but money don comot for my account."],
        "Hausa": ["Na tura {amt} amma bai isa ba, kuma an cire kudin daga asusuna.", "An cire min {amt} amma wanda na tura masa bai samu ba."],
        "Yoruba": ["Mo fi {amt} ranse, won ti yo owo mi sugbon ko de odo eni ti mo fi ranse si.", "Owo {amt} ti mo transfer ko de."],
        "Igbo": ["Ezigara m {amt} ma o rughi onye m zigara ya, ma ewepuola ego n'akaụntụ m.", "Ewepuru {amt} ma transfer ahu erughi."],
    },
    "reversal_delayed": {
        "English": ["My failed transfer of {amt} has not been reversed after three days.", "Still waiting for reversal of {amt}."],
        "Pidgin": ["Una never reverse my {amt} wey fail since three days.", "Reversal of {amt} never come till now."],
        "Hausa": ["Ba a mayar min da {amt} ba tun kwana uku.", "Har yanzu ba a dawo min da {amt} da ya fadi ba."],
        "Yoruba": ["Won ko tii da {amt} mi pada lati ojo meta.", "Mi o tii ri reversal {amt} gba."],
        "Igbo": ["Ha alaghachibeghi {amt} m kemgbe ubochi ato.", "Reversal nke {amt} abiabeghi."],
    },
    "wrong_beneficiary": {
        "English": ["I sent {amt} to a wrong account by mistake, please help recover it."],
        "Pidgin": ["I mistakenly send {amt} enter wrong account, abeg help me collect am back."],
        "Hausa": ["Na tura {amt} zuwa wani asusu bisa kuskure."],
        "Yoruba": ["Mo fi {amt} ranse si account ti ko to nipa asise."],
        "Igbo": ["Ezigara m {amt} n'akaụntụ ezighi ezi."],
    },
    "ussd_session_timeout": {
        "English": ["USSD keeps timing out before I finish the transaction.", "The USSD code shows session ended every time."],
        "Pidgin": ["USSD dey cut before I finish, e dey talk session ended.", "Una USSD no dey work, e just dey timeout."],
        "Hausa": ["USSD yana katsewa kafin in gama.", "Lambar USSD tana cewa session ya kare."],
        "Yoruba": ["USSD n ja ki n to pari.", "Koodu USSD n so pe session ti pari."],
        "Igbo": ["USSD na-akwusi tupu m mechaa.", "USSD na-ekwu na session agwula."],
    },
    "ussd_wrong_charges": {
        "English": ["I was charged {amt} for USSD sessions that failed."], "Pidgin": ["Una charge me {amt} for USSD wey no work."],
        "Hausa": ["An caje ni {amt} saboda USSD da bai yi aiki ba."], "Yoruba": ["Won gba {amt} lowo mi fun USSD ti ko sise."],
        "Igbo": ["Ha ji USSD na-adighi aru oru chaja m {amt}."],
    },
    "atm_debited_no_cash": {
        "English": ["The ATM debited me {amt} but did not dispense cash.", "ATM did not pay out {amt} but I was debited."],
        "Pidgin": ["ATM debit me {amt} but e no bring money.", "I try withdraw {amt}, ATM no pay but alert don enter."],
        "Hausa": ["ATM ta cire min {amt} amma bata bayar da kudi ba.", "Na cire {amt} a ATM, kudin bai fito ba."],
        "Yoruba": ["ATM yo {amt} lowo mi sugbon ko fun mi lowo.", "Mo gba {amt} ni ATM, owo ko jade."],
        "Igbo": ["ATM wepuru {amt} ma o nyeghi m ego.", "Achoro m iwepu {amt} na ATM, ego apughi."],
    },
    "card_declined": {
        "English": ["My card keeps getting declined online and at POS.", "Card declined even though I have funds."],
        "Pidgin": ["My card dey decline for everywhere.", "Card no gree work, e dey talk declined."],
        "Hausa": ["Katina ana kin karbarsa ko ina.", "Katina bata aiki duk da ina da kudi."],
        "Yoruba": ["Won n ko kaadi mi nibi gbogbo.", "Kaadi mi ko sise bo tile je pe mo ni owo."],
        "Igbo": ["A na-ajụ kaadị m ebe niile.", "Kaadị m anaghi aru oru."],
    },
    "card_blocked": {
        "English": ["My card was blocked without notice."], "Pidgin": ["Una block my card without telling me."],
        "Hausa": ["An toshe katina ba tare da sanarwa ba."], "Yoruba": ["Won ti di kaadi mi lai so fun mi."],
        "Igbo": ["Emechiri kaadị m n'agwaghi m."],
    },
    "pos_debited_no_value": {
        "English": ["I paid {amt} at a POS terminal, I was debited but the merchant did not get value."],
        "Pidgin": ["I pay {amt} for POS, dem debit me but the merchant no see am."],
        "Hausa": ["Na biya {amt} ta POS, an cire amma dan kasuwa bai samu ba."],
        "Yoruba": ["Mo san {amt} lori POS, won yo owo sugbon onisowo ko ri i gba."],
        "Igbo": ["Akwuru m {amt} na POS, ewepuru ego ma onye ahia enwetaghi ya."],
    },
    "app_login_failure": {
        "English": ["I cannot log in to the mobile app since the update.", "The app says login failed every time."],
        "Pidgin": ["I no fit login for una app since una update am.", "App dey talk login failed."],
        "Hausa": ["Ba zan iya shiga app din ba tun da aka sabunta.", "App yana cewa shiga ya kasa."],
        "Yoruba": ["Mi o le wole si app lati igba ti won update re.", "App n so pe login ko sise."],
        "Igbo": ["Enweghi m ike ịbanye n'app kemgbe update.", "App na-ekwu na login adaala."],
    },
    "otp_not_received": {
        "English": ["I am not receiving the OTP to complete transactions on the app.", "OTP never arrives, I cannot transfer."],
        "Pidgin": ["OTP no dey come at all, I no fit transfer.", "Una OTP no dey land for my phone."],
        "Hausa": ["Ba na samun OTP domin kammala ciniki.", "OTP baya zuwa, ba zan iya tura kudi ba."],
        "Yoruba": ["Mi o ri OTP gba lati pari idunadura.", "OTP ko de, mi o le transfer."],
        "Igbo": ["Anaghị m enweta OTP iji mechaa azụmahịa.", "OTP anaghi abia, enweghi m ike izipu ego."],
    },
    "app_crash": {
        "English": ["The app crashes as soon as I open it."], "Pidgin": ["App dey close by itself once I open am."],
        "Hausa": ["App yana rufewa da zarar na bude shi."], "Yoruba": ["App n pa ara re ni kete ti mo ba si i."],
        "Igbo": ["App na-emechi ozugbo m mepere ya."],
    },
    "bvn_nin_update": {
        "English": ["I have updated my BVN and NIN but my account still shows incomplete KYC."],
        "Pidgin": ["I don update my BVN and NIN but una still dey talk say my KYC no complete."],
        "Hausa": ["Na sabunta BVN da NIN amma har yanzu asusuna yana nuna KYC bai cika ba."],
        "Yoruba": ["Mo ti se imudojuiwon BVN ati NIN sugbon account mi si n fi KYC ti ko pe han."],
        "Igbo": ["Emeela m BVN na NIN ma akaụntụ m ka na-egosi na KYC ezughi."],
    },
    "account_restricted": {
        "English": ["My account has been restricted and I cannot withdraw."], "Pidgin": ["Una restrict my account, I no fit withdraw."],
        "Hausa": ["An takaita asusuna, ba zan iya cire kudi ba."], "Yoruba": ["Won ti di account mi, mi o le gba owo."],
        "Igbo": ["Egbochiri akaụntụ m, enweghi m ike iwepu ego."],
    },
    "unauthorised_debit": {
        "English": ["{amt} was debited from my account without my authorisation.", "I see a debit of {amt} I did not make."],
        "Pidgin": ["Somebody remove {amt} from my account, no be me do am.", "I see debit of {amt} wey I no know."],
        "Hausa": ["An cire {amt} daga asusuna ba tare da izinina ba."], "Yoruba": ["Won yo {amt} lati inu account mi lai fi ase mi."],
        "Igbo": ["Ewepuru {amt} n'akaụntụ m n'enyeghi m ikike."],
    },
    "sms_charges": {
        "English": ["You charged me {amt} for SMS alerts this month, it is too much."], "Pidgin": ["Una charge me {amt} for SMS alert, e too much."],
        "Hausa": ["An caje ni {amt} na sakon SMS a wannan wata."], "Yoruba": ["Won gba {amt} fun SMS alert losu yii."],
        "Igbo": ["Ha chajara m {amt} maka SMS n'onwa a."],
    },
    "maintenance_fee": {
        "English": ["I was charged a maintenance fee of {amt} on a savings account."], "Pidgin": ["Una remove {amt} as maintenance fee for my savings."],
        "Hausa": ["An cire min {amt} kudin kulawa."], "Yoruba": ["Won yo {amt} gege bi owo itoju."], "Igbo": ["Ewepuru {amt} dika ugwo nlekota."],
    },
    "loan_deduction": {
        "English": ["{amt} was deducted for a loan I already repaid."], "Pidgin": ["Una deduct {amt} for loan wey I don pay finish."],
        "Hausa": ["An cire {amt} na bashin da na riga na biya."], "Yoruba": ["Won yo {amt} fun gbese ti mo ti san tan."],
        "Igbo": ["Ewepuru {amt} maka ego mbinye m kwuchara."],
    },
    "long_wait_time": {
        "English": ["I waited over an hour on the phone and nobody answered."], "Pidgin": ["I don wait for line pass one hour, nobody pick."],
        "Hausa": ["Na jira sama da awa daya a waya ba wanda ya amsa."], "Yoruba": ["Mo duro lori ero ibanisoro ju wakati kan lo."],
        "Igbo": ["Echere m ihe kariri otu awa n'ekwentị, onweghi onye zara."],
    },
    "staff_attitude": {
        "English": ["The staff at the branch were rude to me."], "Pidgin": ["Una staff for branch rude me well well."],
        "Hausa": ["Ma'aikatan reshen sun yi min rashin kunya."], "Yoruba": ["Awon osise ni eka naa ko bo mi daadaa."],
        "Igbo": ["Ndị oru na alaka ahu kparịrị m."],
    },
}


def naira(rng, issue):
    base = {"atm_debited_no_cash": (5e3, 6e4), "pos_debited_no_value": (2e3, 1.5e5), "sms_charges": (200, 2e3),
            "maintenance_fee": (100, 1e3), "ussd_wrong_charges": (50, 1e3)}.get(issue, (2e3, 5e5))
    v = float(np.exp(rng.uniform(np.log(base[0]), np.log(base[1]))))
    v = round(v, -2) if v > 1000 else round(v, -1)
    return v


def amt_text(rng, v, lang):
    if v >= 1000 and v % 1000 == 0 and rng.random() < .4:
        return f"{int(v // 1000)}k"
    return rng.choice([f"N{v:,.0f}", f"₦{v:,.0f}", f"{v:,.0f} naira"]) if lang == "English" else rng.choice([f"N{v:,.0f}", f"₦{v:,.0f}", f"{int(v // 1000)}k" if v >= 1000 else f"N{v:,.0f}"])


def noisify(rng, s, channel):
    if channel in ("social", "app_review") and rng.random() < .3:
        s = s.replace(" you ", " u ").replace("please", "pls").replace("Please", "Pls")
    if channel == "social" and rng.random() < .25:
        s = s + " " + rng.choice(["😡", "🙄", "!!!", "#fixit", "smh"])
    if rng.random() < .05:
        i = rng.integers(1, max(2, len(s) - 1))
        s = s[:i] + s[i + 1:]                       # dropped character typo
    return s


def compose(rng, issue, lang, amount, channel):
    # code-switching: local-language complaints often fall back to English mid-sentence
    clause_lang = lang
    if lang in ("Hausa", "Yoruba", "Igbo") and rng.random() < .2:
        clause_lang = rng.choice(["English", "Pidgin"])
    clause = rng.choice(CLAUSE[issue][clause_lang])
    if "{amt}" in clause:
        clause = clause.replace("{amt}", amt_text(rng, amount, clause_lang))
    parts = [rng.choice(OPEN[lang]), clause, rng.choice(DEMAND[lang])]
    return noisify(rng, " ".join(p for p in parts if p).strip(), channel)


# ------------------------------------------------------------------------------------------- incidents
def incident_specs(days):
    d = lambda x: int(x * DAY)  # noqa: E731
    return [  # start, ramp hours (to peak), plateau hours, decay hours, peak extra complaints/hour
        dict(name="OTP failure after app 4.2.1 release (Android)", issues={"otp_not_received": .7, "app_login_failure": .3},
             cause={"app_version": "4.2.1", "device_os": "Android"}, start=d(3.40), ramp=6, plateau=20, decay=24, peak=95,
             channels={"app_review": .3, "app_chat": .3, "social": .25, "call_centre": .15},
             event=("app_release", "Mobile app 4.2.1 rolled out to Android", -0.7 * HOUR), owner="Mobile app team"),
        dict(name="Reversals stuck on Switch C", issues={"reversal_delayed": .55, "transfer_debited_not_received": .45},
             cause={"switch": "Switch C"}, start=d(7.55), ramp=10, plateau=30, decay=36, peak=70,
             event=("switch_degradation", "Switch C settlement queue degraded (failed-NIP rate 4.8%)", -1.5 * HOUR), owner="Switch operations"),
        dict(name="ATM cluster in Kano debiting without dispensing", issues={"atm_debited_no_cash": 1.0},
             cause={"region": "Kano", "atm_id": ["ATM-KN-003", "ATM-KN-007", "ATM-KN-011", "ATM-KN-014"]},
             start=d(10.30), ramp=4, plateau=14, decay=10, peak=28,
             event=("atm_firmware", "Cash-recycler firmware pushed to Kano ATM fleet", -2 * HOUR), owner="ATM operations"),
        dict(name="USSD timeouts on Telco 2", issues={"ussd_session_timeout": .85, "ussd_wrong_charges": .15},
             cause={"telco": "Telco 2"}, start=d(13.70), ramp=3, plateau=9, decay=8, peak=60,
             event=("telco_link", "USSD gateway link to Telco 2 flapping", -0.5 * HOUR), owner="Digital channels / USSD"),
        dict(name="Card declines after Card scheme B maintenance", issues={"card_declined": 1.0},
             cause={"card_scheme": "Card scheme B"}, start=d(16.40), ramp=2, plateau=7, decay=6, peak=55,
             event=("card_maintenance", "Card scheme B planned maintenance window", -0.3 * HOUR), owner="Card operations"),
        dict(name="POS debit-without-value in Lagos (one acquirer)", issues={"pos_debited_no_value": 1.0},
             cause={"region": "Lagos", "switch": "Switch E"}, start=d(19.25), ramp=8, plateau=16, decay=20, peak=30,
             event=("acquirer_issue", "Switch E acquiring host latency alarm", -1 * HOUR), owner="Card & POS operations"),
        dict(name="Unauthorised debits via app in Rivers (fraud campaign)", issues={"unauthorised_debit": 1.0},
             cause={"region": "Rivers", "channel": "app_chat"}, start=d(22.60), ramp=12, plateau=18, decay=24, peak=14,
             channels={"app_chat": .55, "call_centre": .3, "social": .15},
             event=None, owner="Fraud team (link to MuleRadar)"),
        dict(name="SMS charge rule error for student accounts", issues={"sms_charges": 1.0},
             cause={"segment": "student"}, start=d(26.20), ramp=8, plateau=24, decay=30, peak=40,
             event=("billing_change", "SMS alert billing rule updated", -3 * HOUR), owner="Billing / product"),
    ]


DECOY_EVENTS = [(1.6, "app_release", "Mobile app 4.2.0 rolled out to iOS"), (5.2, "maintenance", "Core banking EOD job rescheduled"),
                (9.1, "switch_degradation", "Switch A brief latency spike (resolved in 6 min)"), (12.4, "app_release", "Mobile app 4.2.3 rolled out to iOS"),
                (15.8, "maintenance", "Internet banking maintenance 01:00-02:00"), (21.3, "telco_link", "SMS gateway failover test"),
                (24.9, "atm_firmware", "Cash-recycler firmware pushed to Abuja ATM fleet"), (28.2, "card_maintenance", "Card scheme A routine maintenance")]


# ------------------------------------------------------------------------------------------- generator
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--per-day", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--shift-days", type=float, default=0.0, help="shift every incident start by up to +/- this many days (held-out sets)")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    reg_names = list(REGIONS)
    reg_w = np.array([REGIONS[r][0] for r in reg_names], float); reg_w /= reg_w.sum()
    issue_names = list(ISSUES)
    iss_w = np.array([ISSUES[i][1] for i in issue_names], float); iss_w /= iss_w.sum()
    ch_names = list(CHANNELS); ch_w = np.array(list(CHANNELS.values())); ch_w /= ch_w.sum()
    seg_names = list(SEGMENTS); seg_w = np.array(list(SEGMENTS.values())); seg_w /= seg_w.sum()
    atms = {r: [f"ATM-{r[:2].upper()}-{i:03d}" for i in range(1, 1 + max(4, int(REGIONS[r][0] * 3)))] for r in reg_names}

    # ---- background arrival times: diurnal x weekly x month-end salary surge (benign) ----
    span = args.days * DAY
    hours = np.arange(args.days * 24)
    hod, dow, dom = hours % 24, (hours // 24) % 7, hours // 24 + 1
    diurnal = np.interp(hod, [0, 5, 8, 11, 14, 18, 21, 23], [.15, .1, .9, 1.4, 1.3, 1.2, .8, .35])
    weekly = np.where(dow == 6, .6, np.where(dow == 5, .85, 1.0))
    salary = np.where((dom >= 25) & (dom <= 28), 1.35, 1.0)
    rate = diurnal * weekly * salary
    lam = rate / rate.sum() * args.per_day * args.days
    counts = rng.poisson(lam)
    ts_bg = np.concatenate([h * HOUR + rng.uniform(0, HOUR, c) for h, c in zip(hours, counts)])

    rows = []

    def draw_app_version(t):
        # adoption: newer versions grow over the month
        w = np.array([1, 2, 3, 4, 5, 4, 2], float)
        if t < 3.4 * DAY: w[5] = 0.3
        if t < 12.4 * DAY: w[6] = 0.1
        return rng.choice(APP_VERSIONS, p=w / w.sum())

    def make(t, issue, incident=-1, forced=None):
        forced = forced or {}
        region = forced.get("region") or rng.choice(reg_names, p=reg_w)
        channel = forced.get("channel") or rng.choice(ch_names, p=ch_w)
        zone = REGIONS[region][1]
        lm = ZONE_LANG[zone]
        langs, lw = list(lm), np.array(list(lm.values()))
        eng_bias = CHANNEL_ENGLISH_BIAS[channel]
        lw = lw * (1 - eng_bias) + np.array([1.0 if l == "English" else 0 for l in langs]) * eng_bias
        lang = rng.choice(langs, p=lw / lw.sum())
        product, _, money = ISSUES[issue]
        amount = naira(rng, issue) if money else 0.0
        is_app = product == "mobile_app" or channel in ("app_chat", "app_review")
        rec = {
            "ts": float(t), "channel": channel, "language": lang, "region": region, "product": product, "issue": issue,
            "app_version": forced.get("app_version") or (draw_app_version(t) if is_app else "n/a"),
            "device_os": forced.get("device_os") or (rng.choice(["Android", "iOS"], p=[.78, .22]) if is_app else "n/a"),
            "switch": forced.get("switch") or (rng.choice(SWITCHES, p=[.3, .25, .2, .15, .1]) if product in ("transfers", "pos") else "n/a"),
            "telco": forced.get("telco") or (rng.choice(TELCOS, p=[.38, .27, .25, .10]) if product == "ussd" else "n/a"),
            "atm_id": (rng.choice(forced["atm_id"]) if isinstance(forced.get("atm_id"), list) else
                       (rng.choice(atms[region]) if issue == "atm_debited_no_cash" else "n/a")),
            "card_scheme": forced.get("card_scheme") or (rng.choice(CARD_SCHEMES, p=[.45, .35, .2]) if product in ("cards", "pos") else "n/a"),
            "segment": forced.get("segment") or rng.choice(seg_names, p=seg_w),
            "amount": amount, "incident_id": incident, "repost_of": -1,
        }
        rec["text"] = compose(rng, issue, lang, amount, channel)
        rows.append(rec)

    for t in ts_bg:
        make(t, rng.choice(issue_names, p=iss_w))

    # ---- incidents ----
    specs = incident_specs(args.days)
    if args.shift_days:
        for sp in specs:
            sp["start"] = int(np.clip(sp["start"] + rng.uniform(-args.shift_days, args.shift_days) * DAY, 3.2 * DAY, (args.days - 3) * DAY))
    incidents, events = [], []
    for k, s in enumerate(specs):
        ramp, plat, dec = s["ramp"] * HOUR, s["plateau"] * HOUR, s["decay"] * HOUR
        grid = np.arange(0, ramp + plat + dec, 600.0)                       # 10-min steps
        shape = np.where(grid < ramp, (grid / ramp) ** 1.5, np.where(grid < ramp + plat, 1.0, np.exp(-(grid - ramp - plat) / (dec / 3))))
        abs_t = s["start"] + grid
        hod_ = (abs_t // HOUR) % 24
        day_shape = np.interp(hod_, [0, 5, 8, 11, 14, 18, 21, 23], [.25, .2, .9, 1.3, 1.3, 1.2, .9, .4])
        n = rng.poisson(s["peak"] / 6 * shape * day_shape)
        iss, iw = list(s["issues"]), np.array(list(s["issues"].values()))
        chs = s.get("channels")
        total = 0
        for t0, c in zip(abs_t, n):
            for t in t0 + rng.uniform(0, 600, c):
                forced = dict(s["cause"])
                if chs:
                    forced.setdefault("channel", rng.choice(list(chs), p=np.array(list(chs.values())) / sum(chs.values())))
                make(t, rng.choice(iss, p=iw / iw.sum()), incident=k, forced=forced)
                total += 1
        peak_t = s["start"] + ramp + plat / 2
        ev_id = None
        if s["event"]:
            ev_id = len(events)
            events.append({"event_id": ev_id, "ts": float(s["start"] + s["event"][2]), "type": s["event"][0],
                           "description": s["event"][1], "causal": True, "incident_id": k})
        incidents.append({"incident_id": k, "name": s["name"], "issues": list(s["issues"]), "cause": s["cause"],
                          "start_ts": float(s["start"]), "peak_ts": float(peak_t), "end_ts": float(s["start"] + ramp + plat + dec),
                          "n_complaints": total, "event_id": ev_id, "owner": s["owner"]})
    for day, typ, desc in DECOY_EVENTS:
        events.append({"event_id": len(events), "ts": float(day * DAY), "type": typ, "description": desc, "causal": False, "incident_id": -1})

    df = pl.DataFrame(rows)

    # ---- decoy 1: a viral social post reposted ~900 times in 6 hours (one signal, not an incident) ----
    viral_t, base = 17.9 * DAY, None
    viral_text = "This bank don finish o, dem no dey answer calls at all. Retweet if you don suffer am too!! #EndBadService"
    for i in range(900):
        t = viral_t + rng.exponential(1.6 * HOUR)
        txt = viral_text if rng.random() < .6 else viral_text.replace("o,", "o!!").replace("Retweet", rng.choice(["RT", "Retweet", "Pls retweet"]))
        rows.append({"ts": float(t), "channel": "social", "language": "Pidgin", "region": rng.choice(reg_names, p=reg_w),
                     "product": "service", "issue": "long_wait_time", "app_version": "n/a", "device_os": "n/a", "switch": "n/a",
                     "telco": "n/a", "atm_id": "n/a", "card_scheme": "n/a", "segment": rng.choice(seg_names, p=seg_w),
                     "amount": 0.0, "incident_id": -2, "repost_of": 0 if i else -1, "text": txt})
    # ---- decoy 2: isolated severe complaints (high value, one-off) ----
    for t in rng.uniform(0, span, 40):
        make(t, rng.choice(["unauthorised_debit", "wrong_beneficiary", "transfer_debited_not_received"]))
        rows[-1]["amount"] = float(rng.choice([2.5e6, 4e6, 7.5e6, 1.2e7]))
        rows[-1]["incident_id"] = -3

    df = (pl.DataFrame(rows).filter(pl.col("ts") < span).sort("ts").with_row_index("complaint_id")
            .with_columns(pl.col("complaint_id").cast(pl.Int64)))
    # repost_of -> id of the first viral post
    first_viral = df.filter((pl.col("incident_id") == -2) & (pl.col("repost_of") == -1))["complaint_id"].min()
    df = df.with_columns(repost_of=pl.when(pl.col("repost_of") == 0).then(first_viral).otherwise(pl.col("repost_of")))
    t0 = np.datetime64("2026-09-01T00:00:00")
    df.write_parquet(out / "complaints.parquet")
    pl.DataFrame(events).sort("ts").write_parquet(out / "events.parquet")
    meta = {"t0": str(t0), "days": args.days, "n_complaints": df.height, "seed": args.seed, "incidents": incidents,
            "decoys": {"viral_repost_storm": {"start_ts": viral_t, "n": 900}, "isolated_high_value": 40,
                       "benign_surge": "month-end salary days 25-28 (+35% all issues)"}}
    (out / "incidents.json").write_text(json.dumps(meta, indent=2))
    print(f"{df.height:,} complaints over {args.days} days -> {out}")
    print(df.group_by("language").len().sort("len", descending=True))
    print(df.group_by("channel").len().sort("len", descending=True))
    for inc in incidents:
        print(f"  incident {inc['incident_id']}: {inc['name']:52s} n={inc['n_complaints']:5d}  start day {inc['start_ts'] / DAY:5.2f}")


if __name__ == "__main__":
    main()
