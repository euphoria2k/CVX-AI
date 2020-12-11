<template>
  <!-- eslint-disable vue/no-use-v-if-with-v-for, vue/no-confusing-v-for-v-if, vue/valid-v-for, vue/no-unused-components -->
  <div>
    <page-title
      :heading="heading"
      :subheading="subheading"
      :icon="icon"
    ></page-title>

    <div class="content">
      <div class="row">
        <div class="col-lg-6">
          <div class="main-card mb-3 card">
            <div class="card-body">
              <div>
                <button
                  id="selection-tooltip"
                  class="btn border-0 btn-transition btn-outline-link"
                  style="margin-left: -10px"
                  v-b-toggle.collapse2
                >
                  <p class="h5">
                    <b-icon icon="search" style="margin-right: 10px"></b-icon
                    >What are you looking for?
                  </p>
                </button>
                <b-tooltip
                  triggers="hover"
                  target="selection-tooltip"
                  placement="top"
                  :delay="{ show: 1000, hide: 0 }"
                  >Click to collapse this segment</b-tooltip
                >

                <b-collapse id="collapse2" visible>
                  <multiselect
                    id="selectColor"
                    v-model="selectedColor"
                    :options="colorOptions"
                    :custom-label="selectColorLabel"
                    placeholder="Select class color (Y8 or Fuzzy)"
                    label="text"
                    track-by="text"
                  ></multiselect>

                  <multiselect
                    id="selectClass"
                    v-model="selectedClass"
                    :options="classOptions"
                    :custom-label="selectClassLabel"
                    placeholder="Select class (person or clothing)..."
                    label="text"
                    track-by="text"
                    style="margin-top: 10px"
                  ></multiselect>

                  <button
                    id="resetSearchButton-tooltip"
                    type="button"
                    class="btn btn-transition btn-block btn-outline-light"
                    style="margin-top: 10px"
                    @click="resetSelect"
                  >
                    Reset search queries
                  </button>
                  <b-tooltip
                    triggers="hover"
                    target="resetSearchButton-tooltip"
                    placement="top"
                    :delay="{ show: 1000, hide: 0 }"
                    >Resets your query selections</b-tooltip
                  >
                  <br />
                  <h5 class="mt-3">Controls:</h5>
                  <br />
                  <button
                    id="setPyDirButton"
                    type="button"
                    class="btn mr-2 mb-2 btn-transition btn-block btn-outline-alternate"
                    @click="getDirectories"
                  >
                    Set flags directories, and configurations
                  </button>
                  <b-tooltip
                    triggers="hover"
                    target="setPyDirButton"
                    placement="top"
                    :delay="{ show: 1000, hide: 0 }"
                    >Opens 4 windows that open consecutively for you to set the
                    directories for the command string that will be used for
                    video analysis</b-tooltip
                  >
                  <button
                    id="generateExec"
                    type="button"
                    class="btn mr-2 mb-2 btn-transition btn-block btn-outline-info"
                    @click="cmd_init"
                  >
                    Generate commands and flags
                  </button>
                  <b-tooltip
                    triggers="hover"
                    target="generateExec"
                    placement="top"
                    :delay="{ show: 1000, hide: 0 }"
                    >Important! Combines flags and directories to be sent to the
                    AI for video analysis</b-tooltip
                  >
                  <button
                    id="findButton-tooltip"
                    type="button"
                    class="btn mr-2 mb-2 btn-transition btn-block btn-outline-secondary"
                    @click="sendExec"
                  >
                    Start video analysis
                  </button>
                  <b-tooltip
                    triggers="hover"
                    target="findButton-tooltip"
                    placement="top"
                    :delay="{ show: 1000, hide: 0 }"
                    >Loads the file and searches for the POI with the search
                    queries, options and flags provided</b-tooltip
                  >
                  <button
                    id="refreshButton-tooltip"
                    type="button"
                    class="btn mr-2 mb-2 btn-transition btn-block btn-outline-danger"
                    @click="showConfirmation"
                  >
                    Restart query
                  </button>
                  <b-tooltip
                    triggers="hover"
                    target="refreshButton-tooltip"
                    placement="top"
                    :delay="{ show: 1000, hide: 0 }"
                    >Start a new fresh query search again</b-tooltip
                  >
                  <b-modal
                    ref="my-modal"
                    hide-footer
                    title="Are you sure you want to restart?"
                  >
                    <div class="d-block text-left" style="margin-bottom: 20px">
                      <p class="h5 d-flex align-items-center">
                        <b-icon
                          icon="exclamation-triangle-fill"
                          variant="danger"
                          font-scale="2"
                          style="margin-right: 10px"
                        ></b-icon
                        >All parsed informaton will be lost!
                      </p>
                    </div>
                    <b-button
                      class="mt-2"
                      variant="outline-success"
                      block
                      @click="refreshThis"
                      >Yes, restart query</b-button
                    >
                    <b-button
                      class="mt-3"
                      variant="outline-secondary"
                      block
                      @click="hideConfirmation"
                      >No, go back</b-button
                    >
                  </b-modal>
                </b-collapse>
                <div class="divider"></div>

                <div class id="showOnClickFind" v-if="onFind">
                  <button
                    class="btn border-0 btn-transition btn-outline-link"
                    style="margin-left: -10px"
                    v-b-toggle.collapse3
                  >
                    <p class="h5">
                      <b-icon
                        icon="file-person"
                        style="margin-right: 10px"
                      ></b-icon
                      >Results
                    </p>
                  </button>
                  <!-- <b-collapse id="collapse3" visible>
                    <ul class="list-group">
                      <li class="list-group-item">
                        <h5 class="list-group-item-heading">
                          Match Confidence:
                        </h5>
                        <div class="progress" style="height: 2rem">
                          <div
                            :style="{ width: flaskConfPercentage }"
                            :aria-valuenow="progressBar_current"
                            role="mb-2 progressbar"
                            aria-valuemin="0"
                            aria-valuemax="100"
                            class="progress-bar bg-alternate"
                          >
                            {{ flaskConfPercentage }}
                          </div>
                        </div>
                      </li>
                      <li class="list-group-item">
                        <h5 class="list-group-item-heading">
                          Matching Frame(s): {{ flaskFrameData }}
                        </h5>
                      </li>
                      <li class="list-group-item">
                        <h5 class="list-group-item-heading">
                          List group item heading
                        </h5>
                        <p class="list-group-item-text">
                          Donec id elit non mi porta gravida at eget metus.
                          Maecenas sed diam eget risus varius blandit.
                        </p>
                      </li>
                    </ul>
                  </b-collapse> -->
                </div>
              </div>
            </div>
          </div>
        </div>

        <div class="col-lg-6">
          <!-- <div class="main-card mb-3 card">
            <span
              id="videoPlayer-tooltip"
              style="margin-left: 22px; margin-top: 22px"
            >
              <p class="h5">
                <b-icon
                  icon="camera-video"
                  style="margin-right: 10px"
                  font-scale="1.3"
                ></b-icon
                >Video Player
              </p>
            </span>
            <b-tooltip
              triggers="hover"
              target="videoPlayer-tooltip"
              placement="top"
              :delay="{ show: 1000, hide: 0 }"
              >A video highlighting your selected queries will be shown
              here</b-tooltip
            >
            <div class="divider" />
            <div class="responsive" v-if="videoData.length > 0"> -->

          <div class="main-card mb-3 card">
            <button
              id="stupidFrameDataViewer-tooltip"
              class="btn border-0 btn-transition btn-outline-link text-left"
              style="margin-left: 10px; margin-top: 20px; margin-bottom: 10px"
              v-b-toggle.collapse4
            >
              <p class="h5 mb-2">
                <b-icon icon="folder-fill" style="margin-right: 10px"></b-icon
                >Directory Viewer
              </p>
            </button>
            <b-tooltip
              triggers="hover"
              target="stupidFrameDataViewer-tooltip"
              placement="top"
              :delay="{ show: 1000, hide: 0 }"
              >View the frames from the video result in detail here</b-tooltip
            >
            <b-collapse id="collapse4" visible>
              <div id="frameDataViewerTab">
                <ejs-filemanager
                  id="file-manager"
                  :contextMenuSettings="contextMenuSettings"
                  :ajaxSettings="ajaxSettings"
                  :height="heightEJS"
                  :toolbarSettings="toolbarSettings"
                >
                </ejs-filemanager>
              </div>
            </b-collapse>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import PageTitle from "../Layout/Components/PageTitle.vue";
import Vue from "vue";
import { FileManagerPlugin, Toolbar } from "@syncfusion/ej2-vue-filemanager";
import { BootstrapVue, BootstrapVueIcons } from "bootstrap-vue";
import Multiselect from "vue-multiselect";

Vue.use(BootstrapVue);
Vue.use(BootstrapVueIcons);
Vue.use(FileManagerPlugin);

export default {
  components: {
    PageTitle,
    Multiselect,
  },
  data() {
    return {
      heading: "Video Footage Playback",
      subheading: "Video footage from selected file will be played here",
      icon: "pe-7s-stopwatch icon-gradient bg-amy-crisp",

      heightEJS: "600px",
      ajaxSettings: {
        // Replace the hosted port number in the place of "{port}"
        url: "http://localhost:8090/",
        getImageUrl: "http://localhost:8090/GetImage",
        uploadUrl: "http://localhost:8090/Upload",
        downloadUrl: "http://localhost:8090/Download",
      },
      contextMenuSettings: {
        file: ["Open", "|", "Details"],
        folder: ["Open", "|", "Details"],
        layout: ["SortBy", "View", "Refresh", "|", "Details", "|"],
        visible: true,
      },
      toolbarSettings: {
        items: ["Refresh", "Details"],
        visible: true,
      },

      exec_py: "",
      // arg1: "",
      arg2: "",
      // arg3: "",
      arg4: "",
      // arg5: "",
      arg6: "",
      // arg7: "",

      //Dropdown box for class flag
      selectedClass: null,
      classOptions: [
        { value: "--person", text: "Person" },
        { value: "--shirt", text: "Shirt" },
        { value: "--jacket", text: "Jacket" },
        { value: "--suit", text: "Suit" },
        { value: "--trouser", text: "Trousers" },
        { value: "--jeans", text: "Jeans" },
        { value: "--dress", text: "Dress" },
        { value: "--skirt", text: "Skirt" },
        { value: "--footwear", text: "Footwear" },
      ],

      //Dropdown box for color flag
      selectedColor: null,
      colorOptions: [
        { value: "--red", text: "[Y8S] Red" },
        { value: "--orange", text: "[Y8S] Orange" },
        { value: "--yellow", text: "[Y8S] Yellow" },
        { value: "--green", text: "[Y8S] Green" },
        { value: "--blue", text: "[Y8S] Blue" },
        { value: "--violet", text: "[Y8S] Violet" },
        { value: "--white", text: "[Y8S] White" },
        { value: "--black", text: "[Y8S] Black" },

        { value: "--Fuzzy_pink", text: "[X11] Pink" },
        { value: "--Fuzzy_red", text: "[X11] Red" },
        { value: "--Fuzzy_orange", text: "[X11] Orange" },
        { value: "--Fuzzy_yellow", text: "[X11] Yellow" },
        { value: "--Fuzzy_brown", text: "[X11] Brown" },
        { value: "--Fuzzy_green", text: "[X11] Green" },
        { value: "--Fuzzy_cyan", text: "[X11] Cyan" },
        { value: "--Fuzzy_blue", text: "[X11] Blue" },
        { value: "--Fuzzy_purple", text: "[X11] Purple" },
        { value: "--Fuzzy_white", text: "[X11] White" },
        { value: "--Fuzzy_black", text: "[X11] Black" },
      ],

      videoData: "",
      onFind: false,

      flaskFrameData: "",
      flaskConfPercentage: "",
      progressBar_current: "",
    };
  },
  provide: {
    filemanager: [Toolbar],
  },

  methods: {
    selectClassLabel({ value, text }) {
      return `${text} (${value})`;
    },
    selectColorLabel({ value, text }) {
      return `${text} (${value})`;
    },

    cmd_init() {
      var condaSet_py = {
        exec: "python",
      };
      // var condaSet_pyDir = {
      //   arg1: "C:/Users/HP/Documents/CVX_AI_WebUI/root/api/object_tracker.py",
      // };
      var condaSet_weight = {
        arg2: "-weights",
      };
      // var condaSet_weightDir = {
      //   arg3:
      //     "C:/Users/HP/Documents/CVX_AI_WebUI/root/api/checkpoints/lastweights",
      // };
      var condaSet_video = {
        arg4: "-video",
      };
      // var condaSet_videoDir = {
      //   arg5:
      //     "C:/Users/HP/Documents/CVX_AI_WebUI/root/api/data/video/samplevideo.mp4",
      // };
      var condaSet_output = {
        arg6: "--output",
      };
      // var condaSet_outputDir = {
      //   arg7:
      //     "C:/Users/HP/Documents/CVX_AI_WebUI/root/api/data/output/success.mp4",
      // };

      var colorFlag = JSON.stringify(this.selectedColor);
      var colorFlag_serialize = JSON.parse(colorFlag);
      delete colorFlag_serialize.text;

      var classFlag = JSON.stringify(this.selectedClass);
      var classFlag_serialize = JSON.parse(classFlag);
      delete classFlag_serialize.text;
      // var colorFlag = JSON.stringify(this.selectedColor);
      // var colorFlag_serialize = JSON.parse(colorFlag);
      // var condaSet_color = colorFlag_serialize.value;

      // var classFlag = JSON.stringify(this.selectedClass);
      // var classFlag_serialize = JSON.parse(classFlag);
      // var condaSet_class = classFlag_serialize.value;

      // this.executeThis =
      //   condaSet_py +
      //   condaSet_weight +
      //   condaSet_video +
      //   condaSet_output +
      //   condaSet_class + " " +
      //   condaSet_color;
      // console.log(this.executeThis);
      console.log(condaSet_py);
      // console.log(condaSet_pyDir);
      console.log(condaSet_weight);
      // console.log(condaSet_weightDir);
      console.log(condaSet_video);
      // console.log(condaSet_videoDir);
      console.log(condaSet_output);
      // console.log(condaSet_outputDir);
      console.log(colorFlag_serialize);
      console.log(classFlag_serialize);

      this.exec_py = condaSet_py;
      // this.arg1 = condaSet_pyDir;
      this.arg2 = condaSet_weight;
      // this.arg3 = condaSet_weightDir;
      this.arg4 = condaSet_video;
      // this.arg5 = condaSet_videoDir;
      this.arg6 = condaSet_output;
      // this.arg7 = condaSet_outputDir;
      this.arg8 = colorFlag_serialize;
      this.arg9 = classFlag_serialize;
    },

    async sendExec() {
      // {python}
      await fetch("http://127.0.0.1:5000/execute", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(this.exec_py),
      })
        // {python file}
        // .then(await this.$http.post("http://127.0.0.1:5000/arg1"))

        // {weights}
        .then(
          await fetch("http://127.0.0.1:5000/arg2", {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify(this.arg2),
          })
        )
        // {weights location}
        // .then(await this.$http.post("http://127.0.0.1:5000/arg3"))

        // {video}
        .then(
          await fetch("http://127.0.0.1:5000/arg4", {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify(this.arg4),
          })
        )
        // {video location}
        // .then(await this.$http.post("http://127.0.0.1:5000/arg5"))

        // {output}
        .then(
          await fetch("http://127.0.0.1:5000/arg6", {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify(this.arg6),
          })
        )
        // {output location}
        // .then(await this.$http.post("http://127.0.0.1:5000/arg7"))

        // {color}
        .then(
          await fetch("http://127.0.0.1:5000/arg8", {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify(this.arg8),
          })
        )
        // {class}
        .then(
          await fetch("http://127.0.0.1:5000/arg9", {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify(this.arg9),
          })
        )
        //combine arguments to one string and call to execute the command
        .then(await this.$http.post("http://127.0.0.1:5000/cvx_start"));
    },

    playVideo: function (event) {
      var input = event.target;
      if (input.files && input.files[0]) {
        var reader = new FileReader();
        reader.onload = (e) => {
          this.videoData = e.target.result;
        };
        reader.readAsDataURL(input.files[0]);
      }
    },

    async getDirectories() {
      await this.$http.post("http://127.0.0.1:5000/setPythonDir");
    },

    showOnFind() {
      this.onFind = true;
    },
    resetSelect() {
      this.selectedClass = null;
      this.selectedColor = null;
    },
    showConfirmation() {
      this.$refs["my-modal"].show();
    },
    hideConfirmation() {
      this.$refs["my-modal"].hide();
    },
    refreshThis() {
      location.reload();
    },
  },
  // created: {
  // const frameData_Response = await fetch("http://localhost:5000/frameData");
  // const frameData_Object = await frameData_Response.json();
  // this.flaskFrameData = frameData_Object.frameData;
  // console.log("Frame data successfully parsed", this.flaskConfPercentage);
  // console.log(frameData_Object);

  // const confPercentage_Response = await fetch(
  //   "http://localhost:5000/confPercentage"
  // );
  // const confPercentage_Object = await confPercentage_Response.json();
  // this.flaskConfPercentage = confPercentage_Object.confPercentage;
  // console.log(
  //   "Confidence percentage successfully parsed",
  //   this.flaskConfPercentage
  // );
  // this.progressBar_current = this.flaskConfPercentage;
  // this.progressBar_current.slice(0, -1);
  // },
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
@import "../../node_modules/vue-multiselect/dist/vue-multiselect.min.css";
</style>