import { CircularProgressbarWithChildren } from "react-circular-progressbar";

  function CircleProgress(props) {
    return (
      <div style={{ marginBottom: 80 }}>
        <div style={{ marginTop: 30, display: "flex" }}>
          <div style={{ width: "30%", paddingRight: 30 }}>{props.children}</div>
          <div style={{ width: "70%" }}>
            <h3 className="h5"></h3>
            <p>{props.description}</p>
          </div>
        </div>
      </div>
    );
};



export default CircleProgress;