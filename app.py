import streamlit as st
import numpy as np
import plotly.graph_objects as go
import time

st.set_page_config(layout="wide")

st.markdown("""
<style>

body{
background:linear-gradient(120deg,#89f7fe,#66a6ff);
}

.title{
font-size:50px;
font-weight:700;
text-align:center;
}

.tank{
width:220px;
height:420px;
border:6px solid black;
border-radius:20px;
margin:auto;
position:relative;
overflow:hidden;
background:white;
}

.water{
position:absolute;
bottom:0;
width:100%;
background:linear-gradient(#4facfe,#00f2fe);
animation:wave 2s infinite linear;
}

@keyframes wave{
0%{transform:translateX(0)}
100%{transform:translateX(-50px)}
}

</style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>💧 Water Tank Simulation</div>",unsafe_allow_html=True)

# sidebar
st.sidebar.header("Parameters")

radius = st.sidebar.slider("Radius",1.0,5.0,2.0)
height_max = st.sidebar.slider("Tank Height",2.0,10.0,5.0)

qin = st.sidebar.slider("Inflow",0.0,0.05,0.02)
qout = st.sidebar.slider("Outflow",0.0,0.05,0.01)

h0 = st.sidebar.slider("Initial Level",0.0,height_max,1.0)

simulate = st.sidebar.button("Run Simulation")

# model
A = np.pi*radius**2

dt = 1
time_max = 100

t = np.arange(0,time_max,dt)
h = np.zeros(len(t))
h[0]=h0

for i in range(1,len(t)):

    dh = (qin-qout)/A
    h[i]=h[i-1]+dh

    h[i]=max(0,min(height_max,h[i]))

col1,col2 = st.columns(2)

tank_placeholder = col1.empty()
chart_placeholder = col2.empty()

if simulate:

    for i in range(len(t)):

        percent = (h[i]/height_max)*100

        tank_html=f"""
        <div class="tank">
        <div class="water" style="height:{percent}%"></div>
        </div>

        <center>
        <h2>{h[i]:.2f} m</h2>
        </center>
        """

        tank_placeholder.markdown(tank_html,unsafe_allow_html=True)

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=t[:i+1],
            y=h[:i+1],
            mode="lines",
            line=dict(width=4)
        ))

        fig.update_layout(
        height=500,
        template="plotly_dark",
        yaxis=dict(range=[0,height_max]),
        xaxis_title="Time",
        yaxis_title="Water Height"
        )

        chart_placeholder.plotly_chart(fig,use_container_width=True)

        time.sleep(0.05)