from __future__ import annotations
import html, math
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binom
import 问题2_求解 as q2

ROOT=Path(__file__).resolve().parent; RESULTS=ROOT/"results"; FIGURES=ROOT/"figures"
YEAR_S=365.25*86400; SAT_COST=5e6; LAUNCH_COST=2e8; BATCH=60
C={"blue":"#0F4D92","teal":"#42949E","red":"#B64342","gray":"#767676","green":"#4C956C"}

def setup():
    plt.rcParams.update({"font.sans-serif":["Microsoft YaHei","SimHei","Arial","DejaVu Sans"],"axes.unicode_minus":False,"svg.fonttype":"none","axes.spines.right":False,"axes.spines.top":False,"legend.frameon":False})
def save(fig,name):
    fig.savefig(FIGURES/f"{name}.svg",bbox_inches="tight"); fig.savefig(FIGURES/f"{name}.png",dpi=300,bbox_inches="tight"); plt.close(fig)
def config():
    x=pd.read_csv(RESULTS/"问题2_推荐构型.csv").iloc[0]
    return q2.Constellation(int(x.planes),int(x.sats_per_plane),float(x.inclination_deg),int(x.phase_factor),float(x.raan0_deg),float(x.phase0_deg))

def risk_table(S):
    # 后五列均是缺失参数的情景值，不是题面事实。
    specs=[("低风险",10,8,10,.003,3,.005),("基准",20,10,20,.005,6,.010),("高风险",40,14,40,.010,12,.020)]
    out=[]
    for name,area,v,conj,trigger,h,other in specs:
        lam=1e-8*area*1e-6*v*YEAR_S; p=1-math.exp(-lam); nm=conj*trigger*.95*.98
        cap=.5*(1-math.exp(-nm*h/(365.25*24))); residual=lam*(1-.95*.99)
        q=1-math.exp(-(residual+other)*7/365.25)
        out.append(dict(scenario=name,area_m2_assumed=area,relative_speed_km_s_assumed=v,conjunctions_per_sat_year_assumed=conj,maneuver_trigger_ratio_assumed=trigger,maneuver_duration_h_assumed=h,other_failure_rate_per_year_assumed=other,single_sat_collision_probability_year=p,constellation_collision_probability_year=1-math.exp(-S*lam),expected_maneuvers_year=S*nm,capacity_loss_ratio=cap,maneuver_cost_5year_yuan=S*nm*2e4*5,residual_collision_rate_year=residual,failure_window_unavailability_q=q))
    return pd.DataFrame(out)

def leave_one_out(cfg):
    _,_,ground,w=q2.make_grid(2.0); times=np.arange(0,q2.SIDEREAL_DAY_S,300.0); threshold=math.cos(q2.PSI_STRICT_RAD)
    loss=np.zeros(cfg.total_satellites); hits=np.zeros(cfg.total_satellites,dtype=int)
    for t in times:
        vis=q2.satellite_unit_vectors(cfg,np.array([t]))[0]@ground.T>=threshold; counts=vis.sum(0); idx=np.flatnonzero(counts==1)
        if len(idx):
            owner=np.argmax(vis[:,idx],axis=0); np.add.at(loss,owner,w[idx]); np.add.at(hits,owner,1)
    loss/=len(times)*w.sum(); ids=np.arange(cfg.total_satellites)
    return pd.DataFrame({"satellite_id":ids,"plane":ids//cfg.sats_per_plane,"slot":ids%cfg.sats_per_plane,"coverage_availability_after_removal":1-loss,"coverage_loss_ratio":loss,"critical_sample_hits":hits})

def reliability(cfg,q):
    return pd.DataFrame([{"spares_per_plane":r,"redundant_satellites":cfg.planes*r,"analytic_system_reliability":binom.cdf(r,cfg.sats_per_plane+r,q)**cfg.planes} for r in range(4)])

def schemes(cfg,risk,rel):
    base=risk[risk.scenario=="基准"].iloc[0]; passing=rel[rel.analytic_system_reliability>=.99]; r=int(passing.iloc[0].spares_per_plane) if len(passing) else 3
    failures=cfg.total_satellites*(base.other_failure_rate_per_year_assumed+base.residual_collision_rate_year)*5; ground=math.ceil(failures+1.645*math.sqrt(failures))
    raw=[("A 每轨在轨备用",cfg.planes*r,cfg.planes*r,float(rel[rel.spares_per_plane==r].analytic_system_reliability.iloc[0]),0,"满足解析99%短期可靠度"),("B 增加1个轨道面",cfg.sats_per_plane,cfg.sats_per_plane,.999,0,"可靠度为代理值，需重新优化相位并几何复核"),("C 地面库存",ground,0,float(rel[rel.spares_per_plane==0].analytic_system_reliability.iloc[0]),7,"不能改善7天窗口，只承担五年补网")]
    rows=[]
    for name,made,orbit,rr,delay,note in raw:
        launch=(math.ceil((cfg.total_satellites+orbit)/BATCH)-math.ceil(cfg.total_satellites/BATCH))*LAUNCH_COST
        replacement=math.ceil(ground/BATCH)*LAUNCH_COST if name.startswith("C") else 0
        rows.append(dict(scheme=name,redundant_satellites=made,initial_orbit_added=orbit,short_term_reliability_proxy=rr,recovery_delay_days=delay,incremental_5year_cost_yuan=made*SAT_COST+launch+replacement+base.maneuver_cost_5year_yuan,note=note))
    return pd.DataFrame(rows)

def plot_all(risk,loo,rel,sch):
    names=[]; colors=[C["green"],C["blue"],C["red"]]
    fig,(a,b)=plt.subplots(1,2,figsize=(12,5)); a.bar(risk.scenario,risk.single_sat_collision_probability_year*1e4,color=colors); a.set_ylabel("单星年碰撞概率 (乘以 1e4)"); b.bar(risk.scenario,risk.constellation_collision_probability_year*100,color=colors); b.set_ylabel("星座一年至少一次碰撞概率 (%)")
    for x,l in [(a,"a"),(b,"b")]: x.text(-.08,1.08,l,transform=x.transAxes,fontsize=22,fontweight="bold")
    fig.tight_layout(); save(fig,"q4_risk_scenarios"); names.append(("q4_risk_scenarios","碎片风险情景"))
    ppm=loo.coverage_loss_ratio*1e6; fig,a=plt.subplots(figsize=(8,5)); a.hist(ppm,bins=30,color=C["blue"]); a.axvline(ppm.max(),color=C["red"],ls="--",label=f"最坏值 {ppm.max():.2f} ppm"); a.set(xlabel="移除单星后的覆盖可用度损失 (ppm)",ylabel="卫星数量"); a.legend(); a.text(-.08,1.08,"a",transform=a.transAxes,fontsize=22,fontweight="bold"); fig.tight_layout(); save(fig,"q4_leave_one_out"); names.append(("q4_leave_one_out","逐星退出脆弱性"))
    fig,a=plt.subplots(figsize=(8,5)); a.plot(rel.redundant_satellites,rel.analytic_system_reliability*100,"o-",color=C["blue"]); a.axhline(99,color=C["red"],ls="--",label="99%要求"); a.set(xlabel="在轨备用卫星数",ylabel="解析系统可靠度 (%)"); a.legend(); a.text(-.08,1.08,"a",transform=a.transAxes,fontsize=22,fontweight="bold"); fig.tight_layout(); save(fig,"q4_redundancy_reliability"); names.append(("q4_redundancy_reliability","冗余与可靠度"))
    labels=["A 在轨","B 加轨道面","C 地面"]; fig,(a,b)=plt.subplots(1,2,figsize=(12,5)); a.bar(labels,sch.incremental_5year_cost_yuan/1e8,color=[C["blue"],C["teal"],C["gray"]]); a.set_ylabel("五年增量成本 (亿元)"); b.bar(labels,sch.short_term_reliability_proxy*100,color=[C["blue"],C["teal"],C["gray"]]); b.axhline(99,color=C["red"],ls="--"); b.set_ylabel("短期可靠度代理值 (%)")
    for x,l in [(a,"a"),(b,"b")]: x.text(-.08,1.08,l,transform=x.transAxes,fontsize=22,fontweight="bold")
    fig.tight_layout(); save(fig,"q4_cost_reliability"); names.append(("q4_cost_reliability","成本—可靠度对比")); return names

def panel(names):
    p=FIGURES/"图表面板.html"; old=p.read_text(encoding="utf-8") if p.exists() else "<!doctype html><meta charset='utf-8'><main></main>"; cards="".join(f'<section><h2>{html.escape(t)}</h2><object data="{n}.svg" type="image/svg+xml"></object></section>' for n,t in names); p.write_text(old.replace("</main>",cards+"</main>"),encoding="utf-8")

def main():
    RESULTS.mkdir(exist_ok=True); FIGURES.mkdir(exist_ok=True); setup(); cfg=config(); print("阶段1/3：风险情景")
    risk=risk_table(cfg.total_satellites); print("阶段2/3：逐星移除"); loo=leave_one_out(cfg)
    base=risk[risk.scenario=="基准"].iloc[0]; rel=reliability(cfg,float(base.failure_window_unavailability_q)); sch=schemes(cfg,risk,rel); print("阶段3/3：方案比较")
    for df,name in [(risk,"问题4_风险情景.csv"),(loo,"问题4_逐星移除.csv"),(rel,"问题4_冗余可靠度.csv"),(sch,"问题4_方案比较.csv")]: df.to_csv(RESULTS/name,index=False,encoding="utf-8-sig")
    lines=["问题四情景分析结果",f"基准单星年碰撞概率: {base.single_sat_collision_probability_year:.6e}",f"基准星座年碰撞概率: {base.constellation_collision_probability_year:.4%}",f"基准年规避次数: {base.expected_maneuvers_year:.2f}",f"平均容量损失: {base.capacity_loss_ratio:.6%}",f"逐星移除最差覆盖可用度: {loo.coverage_availability_after_removal.min():.6%}",f"退出后低于99%的关键卫星数: {(loo.coverage_availability_after_removal<.99).sum()}","注意：截面积、速度、预警率和非碰撞故障率为情景假设；B方案需几何复核。"]
    (RESULTS/"问题4_结果说明.txt").write_text("\n".join(lines),encoding="utf-8"); panel(plot_all(risk,loo,rel,sch)); print("\n".join(lines)); print(sch.to_string(index=False))
if __name__=="__main__": main()
