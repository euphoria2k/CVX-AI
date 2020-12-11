<template>
  <!-- eslint-disable vue/no-use-v-if-with-v-for, vue/no-confusing-v-for-v-if, vue/valid-v-for -->
  <div id="app">
    <page-title
      :heading="heading"
      :subheading="subheading"
      :icon="icon"
    ></page-title>
    <ejs-filemanager
      id="file-manager"
      height="670px"
      :view="view"
      :contextMenuSettings="contextMenuSettings"
      :ajaxSettings="ajaxSettings"
      :toolbarSettings="toolbarSettings"
    >
    </ejs-filemanager>
  </div>
</template>
<script>
import Vue from "vue";
import {
  FileManagerPlugin,
  DetailsView,
  NavigationPane,
  Toolbar,
} from "@syncfusion/ej2-vue-filemanager";
import PageTitle from "../Layout/Components/PageTitle.vue";

Vue.use(FileManagerPlugin);

export default {
  components: {
    PageTitle,
  },
  data() {
    return {
      heading: "CVX-AI Video Archive Repository",
      subheading:
        "All videos in this present in the video directory are shown here.",
      icon: "pe-7s-drawer icon-gradient bg-happy-itmeo",
      // Initializing File Manager NodeJS file system service.
      ajaxSettings: {
        // Replace the hosted port number in the place of "{port}"
        url: "http://localhost:8090/",
        getImageUrl: "http://localhost:8090/GetImage",
        uploadUrl: "http://localhost:8090/Upload",
        downloadUrl: "http://localhost:8090/Download",
      },
      contextMenuSettings: {
        file: ["Details"],
        folder: ["Open", "|", "Details"],
        layout: ["SortBy", "View", "Refresh", "|", "Details", "|"],
        visible: true,
      },
      toolbarSettings: {
        items: ["Refresh", "View", "Details"],
        visible: true,
      },
      view: "Details",
    };
  },
  provide: {
    filemanager: [DetailsView, NavigationPane, Toolbar],
  },
};
</script>
<style>
@import "../../node_modules/@syncfusion/ej2-base/styles/material.css";
@import "../../node_modules/@syncfusion/ej2-icons/styles/material.css";
@import "../../node_modules/@syncfusion/ej2-inputs/styles/material.css";
@import "../../node_modules/@syncfusion/ej2-popups/styles/material.css";
@import "../../node_modules/@syncfusion/ej2-buttons/styles/material.css";
@import "../../node_modules/@syncfusion/ej2-splitbuttons/styles/material.css";
@import "../../node_modules/@syncfusion/ej2-navigations/styles/material.css";
@import "../../node_modules/@syncfusion/ej2-layouts/styles/material.css";
@import "../../node_modules/@syncfusion/ej2-grids/styles/material.css";
@import "../../node_modules/@syncfusion/ej2-vue-filemanager/styles/material.css";
</style>