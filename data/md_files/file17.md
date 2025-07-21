# Numerical Thermal Analysis of Six-Phase PMSMs With Single- and Double-Layer Fractional-Slot Concentrated Windings in Healthy and Faulty Cases

Wessam E. Abdel-Azim [1,2], Alejandro G. Yepes [1], Ahmed Hemeida [3,4], Ayman S. Abdel-Khalik [5],
Shehab Ahmed [6], and Jesús Doval-Gandoy [1]

1 _CINTECX_, _Universidade de Vigo_, _APET_, Vigo, Spain
2
_Department of Electrical Engineering_, _Alexandria University_, Alexandria, Egypt
3
_Department of Electrical Engineering and Automation_, _Aalto University_, Espoo, Finland
4
_Department of Electrical Engineering_, _Cairo University_, Giza, Egypt
5
_Department of Electrical and Computer Engineering_, _Sultan Qaboos University_, Muscat, Oman
6 _CEMSE Division_, _KAUST_, Thuwal, Saudi Arabia
Email: wessam.essam@uvigo.es, agyepes@uvigo.es, ahmed.hemeida@aalto.fi, a.abdelkhalik@squ.edu.om,
shehab.ahmed@kaust.edu.sa, jdoval@uvigo.es



_**Abstract**_ **—Multiphase** **permanent-magnet** **synchronous** **ma-**
**chines (PMSMs) with fractional-slot concentrated windings**
**(FSCWs) are favored for many uninterruptible applications due**
**to their high fault tolerance and torque density. Two different**
**configurations are often adopted: single layer (SL) and double**
**layer (DL). For SL, each slot is filled with one phase winding,**
**while for DL, two phases can share the same slot. Accordingly,**
**many research papers advise adopting SL in fault-tolerant**
**PMSMs, rather than DL, because SL offers higher physical,**
**thermal, and electromagnetic isolation between phases. However,**
**the lower the thermal isolation, as for DL, the greater the**
**heat transfer. This could be expected to reduce the temperature**
**differences between phases, and hence the hot-spot temperatures**
**under unbalanced conditions such as faults. Thus, the established**
**preference of SL over DL for fault-tolerant PMSMs may be ques-**
**tioned, and further analysis is necessary. This paper compares the**
**temperature distribution of six-phase PMSMs between SL and**
**DL FSCWs under both healthy and faulty conditions, considering**
**open-circuit and short-circuit (interturn) winding faults. For**
**accurate results, finite-element analysis and computational fluid**
**dynamics are employed for the loss calculation and thermal**
**simulation, respectively.**
_**Index Terms**_ **—Fault tolerance, fractional-slot concentrated**
**winding, six-phase permanent-magnet synchronous machine,**
**thermal model.**


I. I NTRODUCTION


Multiphase permanent-magnet (PM) synchronous machines
(PMSMs) are versatilely deployed, compared with three-phase
ones, in high-performance and high-reliability applications
such as electric vehicles, ship propulsion, and aerospace due
to their great fault tolerance, reduced torque ripple, better
dc-link utilization, and multiple fault-tolerant (FT) control
strategies [1]. To achieve an efficient PMSM design, the
fractional-slot concentrated winding (FSCW) is adopted in the


This work was supported in part by the Government of Galicia
under the grant GPC-ED431B 2023/12, and in part by the Spanish
State Research Agency MCIN/AEI/10.13039/501100011033/FEDER-UE under projects CNS2022-135773 and PID2022-136908OB-I00.



stator because it offers shorter end-winding length, higher fill
factor and higher torque density compared with distributed
winding [2]. Generally, FSCW can be designed with two different arrangements: single layer (SL) or double layer (DL). In
principle, each slot is occupied with one coil side for SL, while
two coil sides can be included in each slot for DL [2], [3]. SL
produces higher harmonic content in back-electromotive force
(back-EMF) and magnetomotive force (MMF) than DL, and
thereby has a potentially higher overload torque capability than
DL [3]. In addition, the greater electric, magnetic, and thermal
isolation between SL phases is usually considered to make SL
a more suitable candidate for FT PMSMs than DL [2]–[6]. On
the other hand, DL exhibits lower PM eddy-current and core
losses and torque ripple [3].
For safety-critical applications, the fault tolerance should be
ensured in multiphase PMSMs. Among various fault kinds,
open-phase faults (OPFs) and short-circuit faults (SCFs) are
commonly investigated in FT PMSMs [1]. Some fault types
in multiphase drives, e.g., open-/short-circuit switch faults and
high-resistance connections, can be adapted as OPFs [1]. Some
freedom degrees are missed under OPFs depending on the
fault locations, and thereby torque ripple usually increases
if the machine control is not adapted. With FT control, an
enhanced performance can be achieved, in terms of maximizing torque production with ripple-free torque and minimizing
stator copper losses, by optimizing the phase currents [7].
However, some phases having higher rms current values than
others tend to overheat, and hence, the torque capability is
constrained with the thermal limits [8]. Operating PMSMs
beyond their thermal limits can lead to insulation degradation
and an increased risk of PM demagnetization [1]. Thermal
analysis under OPFs is therefore crucial for identifying localized hot-spot regions, enabling the development of effective
FT control strategies that ensure better temperature distribution
and reasonable machine derating [4], [8]–[10].


The thermal behavior of FT PMSMs under SCFs should

also receive a great attention, because of the large resulting
currents and losses. Most severely, one or more turns of a
specific phase may be short-circuited, which is termed as an
inter-turn fault (ITF). The fault current in an ITF is inversely
proportional to the number of shorted turns [11] and depends
on its location, with closer to the slot opening being more
critical [1], [12]. For each phase driven by a separate fullbridge (FB) inverter (commonly adopted in PMSMs [5], [6],

[11]), applying a terminal short circuit (SC) through the two
upper/lower switches is the simplest remedial action to tolerate
ITFs [12]. The circulating SC current produces a flux linkage
opposing the PM flux linkage in the shorted turns [11].
Alternatively, the mutual interaction between the faulty and
healthy phases can be exploited to alleviate the SC current in
the shorted turns. To do so, the healthy-phase currents should
be adapted such that the mutual flux linkage of the healthy
phases nulls the PM flux linkage in the shorted turns [7].
This method is more feasible for DL than for SL since the

mutual coupling between phases is higher [3]. Accordingly,
the remarks in [2]–[7] about the SL isolation being preferred
for FT PMSMs are questionable.
To assess the thermal performance of a machine design,
there are two main approaches, as follows. The well-known
lumped-parameter thermal networks are computationally efficient and offers reasonable accuracy only when thermal parameter estimation and loss calculations are accurate [13]. On the
other hand, numerical methods based on either finite-element
analysis (FEA) or computational fluid dynamics (CFD) give
more precise thermal prediction, just at the cost of increased
computational burden [4], [8].
Many research works have used numerical analysis to assess
the thermal performance of FT PMSMs either with SL [4] or
DL [8] under healthy and faulty conditions. Bianchi _et al._ [5]
perform 2D thermal FEA of a five-phase PMSM with SL
under OPFs while keeping the rated copper losses. To achieve
higher postfault torque, not only the healthy-phase currents
surpass the rated value, but also the hot-spot temperature is
higher than that of rated healthy condition. On the contrary,
Jiang _et al._ [8] evaluate the thermal performance of a fivephase PMSM with DL under minimum-loss (ML) currentcontrol strategy while respecting the maximum temperature
rise of rated healthy operation. In this manner, the healthyphase currents, under various OPF cases, can exceed the rated
machine current to increase the torque, while maintaining the
hot-spot temperature of rated normal operation. Furthermore,
an FT control method with an even temperature distribution
between faulty and healthy modules of a modular PMSM is
proposed in [10]. To achieve this, an electromagnetic-thermal
coupled model is used in the iterative algorithm to generate
the optimum module current. In the same context, the ML and
maximum-torque (MT) current-control strategies are compared
at rated healthy conditions from the thermal point of view
in [4], [9]. It has been found that MT exhibits higher maximum
temperature rise than ML, because higher copper losses are
obtained under MT to achieve the rated healthy torque. All





(a)



(b)



Fig. 1. Cross-sectional view for winding layouts of the FSCW 6PMSMs.
(a) SL. (b) DL.


these works have investigated the temperature fields of multiphase PMSMs with a certain winding design under various FT
control methods without studying the influence of the winding
configuration (SL or DL) on the thermal distribution. Namely,
the thermal coupling between DL phases may play a crucial
role in improving the heat exchange between phases sharing
the same slot. This could affect the temperature distribution
and the hot-spot temperatures substantially, which requires
further research.

In this paper, the thermal behavior of both SL and DL
windings is compared for six-phase PMSMs (6PMSMs) under
different scenarios such as healthy, OPF, and ITF. The CFD
analysis is performed using Ansys Fluent for each studied
case. Furthermore, Ansys Maxwell is used to obtain the loss
model, which provides the heat sources in the thermal model.


II. FSCW 6PMSM D ESIGN


For high-speed applications, a relatively low pole number is typically selected among the possible FSCW slotpole combinations [14]. Since close values of slot and pole
numbers achieve a high fundamental winding factor and a
high torque density [14], the 12-slot/10-pole and 12-slot/14pole combinations are usually considered in the FSCW PMSM
design. Nevertheless, the 12-slot/14-pole PMSM offers better
field-weakening capability, higher power density, and greater
values of maximum speed, compared with the 12-slot/10pole counterpart [15]. Moreover, it also results in reduced
torque ripple and lower material cost [14]. Accordingly, the
12-slot/14-pole FSCW 6PMSM is considered as a case study.
It is designed with a high self-inductance such that the SC
current for a phase SC at rated speed is limited to the rated
machine current. This design approach is favored for FT
PMSMs without interrupted operation under SCFs [6]. The
optimization methodology outlined in [15] is employed to
determine the machine design parameters provided in Table I.
For SL [see Fig. 1(a)], each slot has one coil side with
160 turns _/_ coil. As for DL [see Fig. 1(b)], each slot is filled
with two coil sides either of the same phase or of two different
phases with 80 turns _/_ coil, and thereby the cross-slot mutual
coupling is higher than that in SL layout [3], [6]. For a fair


TABLE I

D ESIGN D ETAILS OF THE FSCW PMSM


Parameter Value Parameter Value

Stator outer diameter (mm) 130 Stator slot number 12
Stator inner diameter (mm) 62 Rotor pole number 14
Airgap length (mm) 1 Turns/slot number 160
PM width (mm) 10 _._ 68 Rated power (kW) 2
Stack length (mm) 60 Rated torque (Nm) 9 _._ 75
Core steel material M235-35 Rated speed (r _/_ min) 2000
PM material N42sh Rated phase voltage (V) 110


comparison, both layouts are wound with a symmetrical sixphase arrangement. As illustrated in [16], asymmetrical configurations are not feasible for SL machines with a 12-slot/14pole topology. It is noteworthy that symmetrical configuration
achieves higher postfault torque than other arrangements [17].
Besides symmetrical winding layout, each phase winding is
fed by an FB inverter so as to separately control each phase [5],

[6], [11].


III. E LECTROMAGNETIC FEA


In this section, the electromagnetic model for both machines
is developed in Ansys Maxwell to acquire the losses, which
will then be required for thermal analysis. Fig. 2 shows
the no-load back-EMF of both windings with trapezoidal
waveforms. The optimum phase currents can be generated
using the generalized strategy in [7]. This strategy is employed
here because it not only minimizes the stator copper loss
with ripple-free torque, but also considers many possibilities
such as nonsinusoidal back-EMF, any number of phases,
different winding connections, various operating modes, and
any fault locations. While retaining most of the previous
features, the current-reference generation method in [18] can
also be adopted if the additional functionalities are desired
such as extending the torque range with minimal torque ripple,
including current and torque-ripple limitations, and enabling
automatic transition from overload to steady-state operation.
The heat sources in PMSMs are mainly stator copper loss
_P_ cu, core loss _P_ core, and PM loss _P_ PM . The loss calculations
adopted in Ansys Maxwell are well-established and detailed
in [4], [8]. All the loss model parameters are listed in Table II.


_A. Healthy Case_


For a fair comparison, the same operating conditions (rated
torque and speed) are applied for both machines. With the
closed-form solution in [7], the phase currents, depicted in
Fig. 3, are obtained for both designs considering the two
different back-EMFs in Fig. 2. It should be emphasized that
no constraint of zero phase-current summation is applied to
the solution of [7], since each phase is driven by an FB
inverter. It can be seen from Fig. 3 that the phase-current rms
value for SL is 3 _._ 68% lower than for DL at the same torque
due to the higher harmonic content in the back-EMF for SL.
Ansys Maxwell is used to obtain the losses of each layout
by exciting the electromagnetic model with the corresponding
phase currents. Table III shows the calculated loss components



(b)


Fig. 2. No-load per-phase back-EMF of the FSCW 6PMSMs at rated speed.
(a) SL. (b) DL.


TABLE II

L OSS M ODEL P ARAMETERS


Parameter Symbol Value

Copper conductivity (S _/_ m) _σ_ cu 58 _·_ 10 [6]

PM conductivity (S _/_ m) _σ_ PM 0 _._ 6 _·_ 10 [6]
Hysteresis loss coefficient _K_ h 172 _._ 042
Eddy-current loss coefficient _K_ c 1 _._ 368
Excess loss coefficient _K_ e 1 _._ 765


of each winding layout. DL exhibits 9 _._ 50% lower total losses
compared with SL. This is mainly attributed to the lower
harmonic components in the DL MMF. It is noteworthy that
the PM eddy-current losses of both layouts are much lower
than the other losses.


_B. OPF Case_


In this case, both machines are simulated under the same
torque and speed (i.e., rated) as in the healthy case, keeping
the machine operation without derating, while phase a is opencircuited ( _i_ a = 0). Similarly, the procedures of the healthy case
are replicated here to obtain the optimal healthy-phase currents
shown in Fig. 4. It is worth highlighting that DL still results
in 9 _._ 24% lower total losses than SL, as indicated in Table III.


_C. ITF Case_


To ensure a fair comparison of the resulting losses in both
machines under the ITF case, the same number of shorted turns
(specifically, 10 turns of phase a) is adopted in both machines.
The fault resistance between shorted turns is assumed to

be 20 mΩ. As recommended in [11], [12], to alleviate the
SC current, an external SC is applied across the faultyphase terminals by turning on the upper/lower switches of the
corresponding FB inverter. The healthy-phase currents are the
same as in the healthy case shown in Fig. 3, but discarding _i_ a .
It is clear from Table III that DL produces higher total losses
than SL by 19 _._ 85% due to the higher resulting fault current















(a)








(a)



(a)



























(b)


Fig. 3. Phase-current references of the FSCW 6PMSMs under the healthy
case at rated torque and speed. (a) SL. (b) DL.


TABLE III

C OMPARISON OF L OSS C OMPONENTS FOR B OTH L AYOUTS AT R ATED

T ORQUE AND S PEED U NDER T HREE D IFFERENT S CENARIOS


_P_ cu (W) _P_ core (W) _P_ PM (W) Total losses (W)
Scenario
SL DL SL DL SL DL SL DL

Healthy 47 _._ 67 47 _._ 01 64 _._ 86 55 _._ 80 7 _._ 00 5 _._ 36 119 _._ 53 108 _._ 17
OPF 58 _._ 77 57 _._ 91 69 _._ 70 61 _._ 10 8 _._ 92 5 _._ 69 137 _._ 39 124 _._ 70

ITF 68 _._ 09 99 _._ 67 53 _._ 10 46 _._ 67 6 _._ 55 6 _._ 76 127 _._ 74 153 _._ 10


in DL compared with SL. This may be due to the mutual flux
linkage of DL healthy phases reinforcing the PM flux linkage
in shorted turns, and further research into the actual causes
may be conducted in the future.


IV. CFD-B ASED T HERMAL A NALYSIS


The loss model provided in Ansys Maxwell is coupled to
the thermal model via Ansys Workbench. The CFD-based
thermal analysis is carried out using Ansys Fluent. The thermal properties of the used materials are given in Table IV.
Regarding the air-gap modeling, a turbulent flow occurs,
when the air-gap Reynolds number _Re_ g is greater than the
critical Reynolds number _Re_ crit (as usual in rotating electric
machines). Otherwise, a laminar flow is considered in the air
gap with the air thermal conductivity [19]. For the turbulent
flow, the air gap is modeled as a solid with an equivalent airgap thermal conductivity _λ_ g, which can be calculated by [4],

[19], [20]



_λ_ g = 0 _._ 069 _η_ _[−]_ [2] _[.]_ [9084] _Re_ [0] g _[.]_ [4614 ln(3] _[.]_ [33361] _[η]_ [)] ;
_Re_ g = _[ω]_ [r] _[R]_ _ν_ [ro] _[δ]_ _> Re_ crit = 41 _._ 2» _Rδ_ si [;] _[ η]_ [ =]



(b)


Fig. 4. Phase-current references of the FSCW 6PMSMs under an OPF in
phase a at rated torque and speed. (a) SL. (b) DL.


the air blown by a fan. Accordingly, the mixed convection
coefficient of the stator surface can be expressed as [4], [8]


_h_ st = 9 _._ 73 + 14 _v_ st [0] _[.]_ [62] (2)


where _v_ st is the air velocity on the stator surface ( _v_ st =
8 m _/_ s), which can be estimated from the rotor operating
speed [21]. The calculated convection coefficient of the stator surface is _h_ st = 60 _._ 55 W _/_ m [2] _/_ K. Since the hot-spot
temperature typically locates at the end windings [4], [10],

[19], heat transfer at end-space surfaces should be taken into
account. The convective heat transfer coefficient of the endspace surface can be computed by [20]


_h_ end = 15 _._ 5 + 4 _._ 495 _v_ end (3)


where _v_ end is the air velocity at the end space, which equals
_ω_ r _R_ ro . From (3), _h_ end = 43 _._ 74 W _/_ m [2] _/_ K.


_A. Healthy Case_


The generated losses of both machines at the same rated
torque and speed from the electromagnetic model are injected
as heat sources in the thermal model to obtain the steadystate temperature distribution. Fig. 5 shows the temperature
distribution of both designs. The maximum temperatures are
located at the end windings. Both layouts exhibit uniform
temperature distributions across phases due to the equal phasecurrent rms values under healthy conditions displayed in
Fig. 3. However, due to the thermal coupling between DL
phases, the DL hot-spot temperature reduces by 9 _._ 02% with
respect to SL, as shown in Fig. 5. To clarify whether the
decrease in hot-spot temperature for DL is due to its lower
losses or to its different thermal conductance across its heat

transfer paths, another simulation is performed. Namely, the
input losses of SL are manually forced to be the same as for
DL in Ansys Fluent, as indicated in Table V. From Table V and
Fig. 6, DL still keeps a lower hot-spot temperature than SL




[ro] _[δ]_

_ν_ _> Re_ crit = 41 _._ 2»



_R_ si



_δ_ si [;] _[ η]_ [ =] _[R]_ _R_ [ro]



(1)

_[R]_ [ro]

_R_ si



where _ω_ r is the angular rotor velocity, _R_ ro is the rotor outer
radius, _R_ si is the stator inner radius, _η_ is th radius ratio, _δ_ is
the air-gap length, and _ν_ is the air kinematic viscosity. The
heat exchange on the stator outer surface is a combination of
natural and forced convection, with the latter resulting from


TABLE IV

T HERMAL M ODEL P ARAMETERS


Material Thermal conductivity Mass density Heat capacity
(W _/_ m _/_ K) (Kg _/_ m [3] ) (J _/_ Kg _/_ K)


Copper 387 _._ 6 8978 381
Insulation 0 _._ 3 2000 502

M235-35 22 7600 460

N42sh 6 _._ 45 7600 460 _._ 55
Air gap 1 _._ 94 1 _._ 1 1006









Fig. 7. Temperature distribution under the OPF case with phase a open at
rated torque and speed. (a) SL. (b) DL.













Fig. 5. Temperature distribution under the healthy case at rated torque and
speed. (a) SL. (b) DL.

















Fig. 6. Temperature distribution under the healthy case for the same input
losses and rated speed. (a) SL. (b) DL.


(by 7 _._ 51%). This reveals that the lower maximum temperature
for DL is mainly due to the heat exchange between the phases
sharing the same slot, and not to its reduction in losses.


_B. OPF Case_


Opening phase a makes it a passive phase, while the other
healthy phases are active ones. The heat transfer between
active and passive phases can be better exploited for DL than
for SL due to the physical sharing of two DL phases within
the same slot. As depicted in Fig. 7, DL outperforms SL by
lowering the maximum temperature by 11 _._ 04%. Moreover, if
the losses of DL are manually forced as equal inputs for both
SL and DL, as shown in Table V and Fig. 8, the maximum
temperature for DL is still lower (by 9 _._ 30%). This confirms
the relation of the DL temperature decrease with the (higher)
thermal conductances of DL between phases, regardless of
the losses. It can also be noted from Fig. 7 that the hot-spot
temperature of both designs is located at phase d, which has
the highest phase-current rms value (see Fig. 4).


_C. ITF Case_


Fig. 9 displays the temperature fields of both machines
under the ITF condition discussed in Section III-C. The hot


Fig. 8. Temperature distribution under the OPF case (with phase a open) for
the same input losses and rated speed. (a) SL. (b) DL.


spot temperature is higher than for the healthy and OPF cases.
Consequently, reducing this temperature is essential to avoid
problems like PM demagnetization, insulation degradation,
and thermal runaway [1]. In contrast to the maximum temperature of OPF case, the faulty phase in ITF case produces
the highest temperature with respect to other healthy phases
due to the excessive SC current in the faulty phase. It can be
observed from Fig. 9 that the hot-spot temperature is higher for
DL than for SL, mainly due to the higher copper losses of DL
compared with SL in this case (see Table III). Nevertheless, if
the losses of SL are manually defined to match the DL losses,
DL showcases a 10 _._ 19% lower hot-spot temperature than SL,
as depicted in Fig. 10 and Table V. This is thanks to the greater
thermal coupling between healthy and faulty phases for DL.


V. C ONCLUSIONS


In this study, the steady-state temperature distribution of
FSCW 6PMSMs with both SL and DL windings is investigated using CFD-based thermal model under the prospective
operating cases (healthy, OPF, and ITF). Under each case,
both layouts are thermally compared twice: one at the same
rated torque and speed, and the other at the same input
losses. Thanks to the reduction of the MMF harmonics for
DL, it achieves reduced total losses by 9 _._ 50% and 9 _._ 24%
under healthy and OPF conditions, respectively, compared
with SL. However, under the ITF case, it generates 19 _._ 85%
higher losses than SL, possibly due to the mutual flux linkage
of healthy phases boosting the PM flux in shorted turns.
Thermally, the hot-spot temperature is notably reduced by
9 _._ 02% and 11 _._ 04% (at the same rated torque and speed)
under healthy and OPF cases, respectively, compared with
SL. Under the ITF case, a much higher hot-spot temperature


