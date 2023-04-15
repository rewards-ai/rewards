import React from "react";
import clsx from "clsx";
import styles from "./styles.module.css";

const FeatureList = [
  {
    title: "Easy to Use",
    Svg: require("@site/static/img/easy_to_use.svg").default,
    description: (
      <>
        Rewards AI simplifies reinforcement learning for developers and
        students. Learn quickly and build high-performing agents without
        worrying about complex environments.
      </>
    ),
  },
  {
    title: "Focus on What Matters",
    Svg: require("@site/static/img/focus_on_what_matters.svg").default,
    description: (
      <>
        With Rewards AI, experiment with RL agents and train them with just a
        few lines of code. Spend more time learning and less time on technical
        details.
      </>
    ),
  },
  {
    title: "Compete and have fun",
    Svg: require("@site/static/img/have_fun_learning.svg").default,
    description: (
      <>
        Join our community and compete on our leaderboard. Showcase your agents
        and have fun while learning. Check out our documentation for more
        information.
      </>
    ),
  },
];

function Feature({ Svg, title, description }) {
  return (
    <div className={clsx("col col--4")}>
      <div className="text--center">
        <Svg className={styles.featureSvg} role="img" />
      </div>
      <div className="text--center padding-horiz--md">
        <h3>{title}</h3>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures() {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
