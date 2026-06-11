import React from "react";
import ReactDOM from "react-dom/client";
import { Amplify } from "aws-amplify";
import { Authenticator, translations } from "@aws-amplify/ui-react";
import { I18n } from "aws-amplify/utils";
import "@aws-amplify/ui-react/styles.css";
import App from "./App.tsx";
import "./index.css";
import outputs from "../amplify_outputs.json";

Amplify.configure(outputs);

// ログイン UI を日本語化
I18n.putVocabularies(translations);
I18n.setLanguage("ja");

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <Authenticator loginMechanisms={["email"]} signUpAttributes={["email"]}>
      {({ signOut, user }) => (
        <App
          signOut={signOut}
          username={user?.signInDetails?.loginId ?? user?.username}
        />
      )}
    </Authenticator>
  </React.StrictMode>
);
