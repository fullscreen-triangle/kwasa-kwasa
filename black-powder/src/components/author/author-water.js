import React from "react";
import WaterWave from "react-water-wave";

class AuthorWater extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      counter: 0,
      backgroundImgs: [
        "img/garden/Triptych_of_Garden_of_Earthly_Delights.jpg",
      ],
    };
  }

  componentDidMount() {
    this.interval = setInterval(() => this.updateCounter(), 3000);
  }

  updateCounter() {
    const { counter, backgroundImgs, showImgs } = this.state;

    this.setState(
      {
        counter: counter + 1,
      },
      () => {
        if (counter === backgroundImgs.length - 1) {
          this.setState({
            counter: 0,
          });
        }
      }
    );
  }

  render() {
    return (
      <div className="author_image">
        <WaterWave
          style={{
            width: `100%`,
            height: `100%`,
            position: `relative`,
            backgroundRepeat: `no-repeat`,
            backgroundSize: `cover`,
            backgroundPosition: `center`,
            // background: `url(${"img/garden/Triptych_of_Garden_of_Earthly_Delights.jpg"}) no-repeat center center fixed`,
          }}
          dropRadius={12}
          perturbance={0.01}
          interactive={true}
        >
          {(methods) => <></>}
        </WaterWave>
      </div>
    );
  }
}
export default AuthorWater;
